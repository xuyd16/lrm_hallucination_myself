# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings
from typing import Dict, Tuple, Optional, List, Union
from torch.nn.init import xavier_uniform_, constant_

warnings.filterwarnings('ignore')

class ModelConfig:
    """重构的模型配置类，适配新的数据格式"""
    
    def __init__(self):
        # 输入参数 - 重构后的格式
        self.num_regions = 5                    # 脑区数量（额叶、中央、顶叶、颞叶、枕叶）
        self.max_electrodes_per_region = 24     # 每个脑区最大电极数
        self.sequence_length = 1600             # 时间序列长度
        self.embed_dim = 768                    # 嵌入维度
        
        # Transformer架构参数
        self.num_heads = 24                      # 注意力头数
        self.depth = 24                          # Transformer层数
        self.mlp_ratio = 4.0                    # MLP隐藏层比例
        
        # 正则化参数
        self.drop_rate = 0.1                    # Dropout率
        self.attn_drop_rate = 0.1              # 注意力Dropout率
        self.drop_path_rate = 0.1              # 随机深度率
        
        # 位置编码参数
        self.use_abs_pos = True                 # 是否使用绝对位置编码
        self.use_rel_pos = False                # 是否使用相对位置编码
        self.use_time_embed = True              # 是否使用时间嵌入
        
        # 掩码参数
        self.mask_ratio = 0.15                  # 时域掩码比例
        self.freq_mask_ratio = 0.3             # 频域掩码比例
        self.mask_strategy = 'random'           # 掩码策略
        self.mask_noise_ratio = 0.005          # 掩码噪声比例
        
        # MoE参数
        self.use_moe = True                     # 是否使用MoE
        self.num_experts = 16                   # 专家数量
        self.top_k_experts = 2                  # 激活的专家数量
        self.moe_aux_loss_coeff = 0.01         # MoE辅助损失系数
        
        # 频域处理参数
        self.n_freq_bands = 5                   # 频带数量
        self.freq_loss_weight = 0.1            # 频域损失权重（从0.3降低到0.1）
        self.time_loss_weight = 0.9            # 时域损失权重（从0.7提高到0.9）
        
        # 训练参数
        self.lr = 1e-4                         # 学习率
        self.weight_decay = 1e-4               # 权重衰减
        self.warmup_epochs = 5                 # 预热轮数
        self.use_amp = True                    # 混合精度训练
        self.clip_grad = 1.0                   # 梯度裁剪
        self.batch_size = 32                   # 批次大小
        
        # 数据标准化参数
        self.use_layer_norm = True             # 是否使用层归一化
        self.use_batch_norm = True             # 是否使用批归一化
        self.eps = 1e-6                       # 数值稳定性参数
        
        # 初始化参数
        self.init_std = 0.02                   # 初始化标准差
        
        # 脑区和电极相关配置
        self.electrode_names = [               # 默认电极名称（用于调试和可视化）
            'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
            'C3', 'C4', 'Cz', 'P3', 'P4', 'P7', 'P8', 'Pz',
            'O1', 'O2', 'Oz', 'T3', 'T4', 'T5', 'T6'
        ]
        
        self.region_names = ['frontal', 'central', 'parietal', 'temporal', 'occipital']
        
        # 调试参数
        self.debug = False                     # 调试模式
        self.freq_eval = True                  # 频域评估


class EEGRegionProjection(nn.Module):
    """
    简化的EEG脑区投影层，替换原来的DualDomainProjection
    将[B, R, E, T]格式的输入转换为Transformer需要的序列格式
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_regions = config.num_regions
        self.max_electrodes_per_region = config.max_electrodes_per_region
        self.sequence_length = config.sequence_length
        self.embed_dim = config.embed_dim
        
        # 时间投影层 - 将时间序列投影到嵌入空间
        self.time_projection = nn.Sequential(
            nn.Conv1d(1, config.embed_dim // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(config.embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(config.embed_dim // 4, config.embed_dim // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.embed_dim // 2, config.embed_dim)
        )
        
        # 频域特征提取
        # 防护检查：确保n_freq_bands至少为1，用于创建有效的参数和层
        effective_n_freq_bands = max(1, config.n_freq_bands)
        self.freq_bands = nn.Parameter(
            torch.linspace(0.5, 50, effective_n_freq_bands + 1)
        )  # 频带边界
        
        # 频域投影层
        self.freq_projection = nn.Sequential(
            nn.Linear(effective_n_freq_bands, config.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.embed_dim // 2, config.embed_dim)
        )
        
        # 原始信号投影（用于重建任务）
        self.raw_projection = nn.Sequential(
            nn.Linear(config.sequence_length, config.embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.drop_rate),
            nn.Linear(config.embed_dim * 2, config.embed_dim)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化投影层权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据 [B, R, E, T]
            padding_mask: 填充掩码 [B, R, E]，True表示填充位置
            
        Returns:
            time_features: 时域特征 [B, R*E, D]
            freq_features: 频域特征 [B, R*E, D] 
            raw_features: 原始信号特征 [B, R*E, D]
        """
        B, R, E, T = x.shape
        
        # 重新塑形为 [B*R*E, T]，将每个电极的时间序列视为独立样本
        x_reshaped = x.view(B * R * E, T)
        
        # 处理padding mask
        if padding_mask is not None:
            mask_reshaped = padding_mask.view(B * R * E)
            # 将填充位置的数据置零
            x_reshaped = x_reshaped * (~mask_reshaped).float().unsqueeze(-1)
        
        # 时域特征提取
        time_features = self.time_projection(x_reshaped.unsqueeze(1))  # [B*R*E, 1, T] -> [B*R*E, D]
        
        # 频域特征提取
        freq_features = self._extract_frequency_features(x_reshaped)  # [B*R*E, D]
        
        # 原始信号投影（用于重建）
        raw_features = self.raw_projection(x_reshaped)  # [B*R*E, D]
        
        # 应用层归一化
        time_features = self.layer_norm(time_features)
        freq_features = self.layer_norm(freq_features)
        raw_features = self.layer_norm(raw_features)
        
        # 重新塑形为序列格式 [B, R*E, D]
        seq_len = R * E
        time_features = time_features.view(B, seq_len, self.embed_dim)
        freq_features = freq_features.view(B, seq_len, self.embed_dim)
        raw_features = raw_features.view(B, seq_len, self.embed_dim)
        
        return time_features, freq_features, raw_features
    
    def _extract_frequency_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取频域特征
        
        Args:
            x: 输入时间序列 [B*R*E, T]
            
        Returns:
            freq_features: 频域特征 [B*R*E, D]
        """
        # 使用float32进行FFT计算以避免cuFFT的half precision限制
        with torch.cuda.amp.autocast(enabled=False):
            x_float = x.float()  # 确保输入为 float32
            
            # 计算FFT
            fft_result = torch.fft.rfft(x_float, dim=-1)
            power_spectrum = torch.abs(fft_result) ** 2
            
            # 计算频率轴
            freqs = torch.fft.rfftfreq(x_float.size(-1), device=x.device) * 250  # 假设采样率250Hz
            
            # 按频带提取特征
            band_powers = []
            # 防护检查：确保n_freq_bands至少为1
            if self.config.n_freq_bands <= 0:
                # 如果频带数为0，创建一个默认的频带特征
                band_power = power_spectrum.mean(dim=-1)  # 使用整个频谱的平均值
                band_powers.append(band_power)
            else:
                for i in range(self.config.n_freq_bands):
                    band_mask = (freqs >= self.freq_bands[i]) & (freqs < self.freq_bands[i + 1])
                    if band_mask.any():
                        band_power = power_spectrum[:, band_mask].mean(dim=-1)
                    else:
                        band_power = torch.zeros(power_spectrum.size(0), device=x.device)
                    band_powers.append(band_power)
            
            # 堆叠频带特征
            band_features = torch.stack(band_powers, dim=-1)  # [B*R*E, n_freq_bands]
            
            # 通过频域投影层（也在float32上下文中）
            freq_features = self.freq_projection(band_features)  # [B*R*E, D]
        
        return freq_features


class CrossAttentionFusion(nn.Module):
    """
    双向交叉注意力融合模块
    - Time features query Frequency features
    - Frequency features query Time features
    - 最终将融合后的两个特征流进行合并
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        drop_rate = config.drop_rate

        # 时间特征 -> 查询Q, 频率特征 -> 键K/值V
        self.t_q_f_kv_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm1_t = nn.LayerNorm(embed_dim)
        self.mlp_t = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.norm2_t = nn.LayerNorm(embed_dim)

        # 频率特征 -> 查询Q, 时间特征 -> 键K/值V
        self.f_q_t_kv_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm1_f = nn.LayerNorm(embed_dim)
        self.mlp_f = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.norm2_f = nn.LayerNorm(embed_dim)

        # 最终融合投影层
        self.final_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        # x_time 和 x_freq 的形状均为 [B, L, D]
        
        # 1. 时间查询频率
        t_norm = self.norm1_t(x_time)
        f_norm = self.norm1_f(x_freq)
        # 时间特征从频率特征中吸取信息
        fused_t, _ = self.t_q_f_kv_attn(query=t_norm, key=f_norm, value=f_norm)
        x_time = x_time + fused_t # 残差连接
        x_time = x_time + self.mlp_t(self.norm2_t(x_time)) # FFN

        # 2. 频率查询时间 (使用原始的x_time作为Key/Value以获得最原始信息)
        t_norm = self.norm1_t(x_time) # 重新norm
        f_norm = self.norm1_f(x_freq)
        # 频率特征从时间特征中吸取信息
        fused_f, _ = self.f_q_t_kv_attn(query=f_norm, key=t_norm, value=t_norm)
        x_freq = x_freq + fused_f # 残差连接
        x_freq = x_freq + self.mlp_f(self.norm2_f(x_freq)) # FFN

        # 3. 最终融合
        # 将两个经过深度交互的特征流拼接起来
        final_fused = torch.cat([x_time, x_freq], dim=-1) # -> [B, L, 2*D]
        # 投影回原始维度
        projected_fused = self.final_proj(final_fused) # -> [B, L, D]
        
        return self.final_norm(projected_fused)


class MoELayer(nn.Module):
    """混合专家层（保持不变，但简化注释）"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.input_dim = input_dim
        self.hidden_dim = int(input_dim * config.mlp_ratio)
        
        # 门控网络
        self.gate = nn.Linear(input_dim, self.num_experts, bias=False)
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.drop_rate),
                nn.Linear(self.hidden_dim, input_dim)
            ) for _ in range(self.num_experts)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate.weight)
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)
        
        # 门控计算
        gate_logits = self.gate(x_flat)  # [B*L, E]
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-K选择
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # 重新归一化
        
        # 计算专家输出
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_indices = topk_indices[:, i]
            expert_probs = topk_probs[:, i].unsqueeze(-1)
            
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_probs[mask] * expert_output
        
        # 计算辅助损失（负载均衡）
        aux_loss = self._compute_aux_loss(gate_probs)
        
        return output.view(batch_size, seq_len, input_dim), aux_loss
    
    def _compute_aux_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """计算负载均衡辅助损失"""
        mean_prob = gate_probs.mean(dim=0)  # [E]
        mean_prob_topk = (gate_probs > gate_probs.topk(self.top_k, dim=-1)[0][..., -1:]).float().mean(dim=0)
        aux_loss = torch.sum(mean_prob * mean_prob_topk) * self.num_experts
        return aux_loss


class RotaryPositionEncoding(nn.Module):
    """旋转位置编码（保持不变）"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 预计算旋转角度
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用旋转位置编码"""
        seq_len = q.shape[-2]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed


class TransformerBlock(nn.Module):
    """Transformer块（保持主要结构，简化注释）"""
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        # 多头注意力
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.attn_drop = nn.Dropout(config.attn_drop_rate)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(config.drop_rate)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.eps)
        
        # 位置编码
        if config.use_rel_pos:
            self.rotary_pos_emb = RotaryPositionEncoding(self.head_dim)
        
        # FFN或MoE
        if config.use_moe:
            self.mlp = MoELayer(config, self.embed_dim)
        else:
            mlp_hidden_dim = int(self.embed_dim * config.mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(self.embed_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.drop_rate),
                nn.Linear(mlp_hidden_dim, self.embed_dim),
                nn.Dropout(config.drop_rate)
            )
            
        # 随机深度
        self.drop_path_rate = config.drop_path_rate * layer_idx / max(config.depth - 1, 1)
            
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, 
               attn_mask: Optional[torch.Tensor] = None,
               key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, L, D = x.shape
        
        # 多头注意力
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # 应用旋转位置编码
        if hasattr(self, 'rotary_pos_emb'):
            q, k = self.rotary_pos_emb.apply_rotary_pos_emb(q, k)
        
        # 计算注意力
        attn_output, aux_loss = self._attention(q, k, v, attn_mask, key_padding_mask)
        
        # 投影和残差连接
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        
        # 随机深度
        if self.training and self.drop_path_rate > 0:
            if torch.rand(1) < self.drop_path_rate:
                attn_output = attn_output * 0
        
        x = residual + attn_output
        
        # MLP
        residual = x
        x = self.norm2(x)
        
        if self.config.use_moe:
            mlp_output, moe_aux_loss = self.mlp(x)
            aux_loss = aux_loss + moe_aux_loss if aux_loss is not None else moe_aux_loss
        else:
            mlp_output = self.mlp(x)
        
        # 随机深度
        if self.training and self.drop_path_rate > 0:
            if torch.rand(1) < self.drop_path_rate:
                mlp_output = mlp_output * 0
        
        x = residual + mlp_output
        
        return x, aux_loss
    
    def _attention(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, H, L, D = q.shape
        scale = D ** -0.5
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 应用掩码
        if key_padding_mask is not None:
            # key_padding_mask: [B, L], True表示需要被掩盖的位置
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, self.embed_dim)
        
        return output, None


class DualDomainTransformerMEM(nn.Module):
    """
    重构的双域Transformer模型
    支持新的[B,R,E,T]输入格式和padding mask
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_regions = config.num_regions
        self.max_electrodes_per_region = config.max_electrodes_per_region
        self.embed_dim = config.embed_dim
        
        # 输入投影层（替换原来的DualDomainProjection）
        self.region_projection = EEGRegionProjection(config)
        
        # (新增) 交叉注意力融合模块
        self.fusion_module = CrossAttentionFusion(config)
        
        # 位置嵌入 - 简化版本
        # 脑区位置嵌入
        self.region_embedding = nn.Embedding(config.num_regions, config.embed_dim)
        # 脑区内电极位置嵌入
        self.intra_region_pos_embedding = nn.Embedding(config.max_electrodes_per_region, config.embed_dim)
        
        # (新增) 序列位置嵌入
        # 序列长度为 R * E = 5 * 24 = 120
        self.positional_embedding = nn.Embedding(config.num_regions * config.max_electrodes_per_region, config.embed_dim)
        
        # (新增) 添加一个可学习的 [MASK] Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.depth)
        ])
        
        # 输出层归一化
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.eps)
        
        # 重建头
        # 时域重建头
        self.time_reconstruction_head1 = nn.Linear(config.embed_dim, config.sequence_length)
        self.time_reconstruction_head2 = nn.Linear(config.embed_dim, config.sequence_length)
        
        # 频域重建头
        # 防护检查：确保n_freq_bands至少为1，用于创建有效的层
        effective_n_freq_bands = max(1, config.n_freq_bands)
        self.freq_reconstruction_head1 = nn.Linear(config.embed_dim, effective_n_freq_bands)
        self.freq_reconstruction_head2 = nn.Linear(config.embed_dim, effective_n_freq_bands)
        
        # Dropout
        self.dropout = nn.Dropout(config.drop_rate)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.init_std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
        # (新增) 对 mask_token 进行初始化
        if hasattr(self, 'mask_token'):
            nn.init.normal_(self.mask_token, std=self.config.init_std)
    
    def forward(self, x: torch.Tensor, 
                time_mask: Optional[torch.Tensor] = None,
                     freq_mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据 [B, R, E, T]
            time_mask: 时域掩码 [B, R*E] (可选)
            freq_mask: 频域掩码 [B, n_freq_bands] (可选)
            padding_mask: 填充掩码 [B, R, E], True表示填充位置
            
        Returns:
            time_pred1: 时域预测1 [B, R*E, T]
            time_pred2: 时域预测2 [B, R*E, T]  
            freq_pred1: 频域预测1 [B, R*E, n_freq_bands]
            freq_pred2: 频域预测2 [B, R*E, n_freq_bands]
            moe_aux_loss: MoE辅助损失
        """
        B, R, E, T = x.shape
        seq_len = R * E
        
        # 1. 投影到特征空间 (并行计算时域和频域特征)
        time_features, freq_features, raw_features = self.region_projection(x, padding_mask)
        # -> 输出形状: [B, 120, D]

        # 2. (核心修改) 使用交叉注意力进行深度融合
        fused_features = self.fusion_module(time_features, freq_features)
        
        # 3. (核心修改) 注入所有结构化嵌入
        # 3a. 空间结构嵌入
        region_ids = torch.arange(R, device=x.device).repeat_interleave(E).unsqueeze(0).expand(B, -1)
        electrode_ids = torch.arange(E, device=x.device).repeat(R).unsqueeze(0).expand(B, -1)
        region_emb = self.region_embedding(region_ids)
        electrode_emb = self.intra_region_pos_embedding(electrode_ids)
        
        # 3b. 序列位置嵌入
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.positional_embedding(position_ids)
        
        # 将融合后的特征与所有嵌入相加
        x_embedded = fused_features + region_emb + electrode_emb + pos_emb
        x_embedded = self.dropout(x_embedded)
        
        # 4. 处理padding mask (保持不变)
        if padding_mask is not None:
            key_padding_mask = padding_mask.view(B, seq_len)
        else:
            key_padding_mask = None
            
        # 5. 掩码自编码应用 (保持不变)
        if time_mask is not None:
            mask = time_mask.unsqueeze(-1)
            x_embedded = torch.where(mask, self.mask_token, x_embedded)

        # 6. 通过Transformer层 (保持不变)
        total_aux_loss = 0
        for layer in self.layers:
            x_embedded, aux_loss = layer(
                x_embedded, 
                attn_mask=None,
                key_padding_mask=key_padding_mask
            )
            if aux_loss is not None:
                total_aux_loss += aux_loss
        
        # 7. 后续步骤 (层归一化和重建) 保持不变
        x_embedded = self.norm(x_embedded)
        time_pred1 = self.time_reconstruction_head1(x_embedded)
        time_pred2 = self.time_reconstruction_head2(x_embedded)
        freq_pred1 = self.freq_reconstruction_head1(x_embedded)
        freq_pred2 = self.freq_reconstruction_head2(x_embedded)
        
        return time_pred1, time_pred2, freq_pred1, freq_pred2, total_aux_loss
    
    def get_targets(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取训练目标"""
        B, R, E, T = x.shape
        
        # 时域目标：直接重塑输入
        time_targets = x.view(B, R * E, T)
        
        # 频域目标：提取频域特征
        x_flat = x.view(B * R * E, T)
        
        # 使用float32进行FFT计算以避免cuFFT的half precision限制
        with torch.cuda.amp.autocast(enabled=False):
            x_flat_float = x_flat.float()  # 确保输入为 float32
            
            # 计算FFT
            fft_result = torch.fft.rfft(x_flat_float, dim=-1)
            power_spectrum = torch.abs(fft_result) ** 2
            
            # 计算频率轴
            freqs = torch.fft.rfftfreq(T, device=x.device) * 250  # 假设采样率250Hz
            
            # 按频带提取特征
            band_powers = []
            # 防护检查：确保n_freq_bands至少为1
            effective_n_freq_bands = max(1, self.config.n_freq_bands)
            if self.config.n_freq_bands <= 0:
                # 如果频带数为0，创建一个默认的频带特征
                band_power = power_spectrum.mean(dim=-1)  # 使用整个频谱的平均值
                band_powers.append(band_power)
            else:
                freq_bands = torch.linspace(0.5, 50, effective_n_freq_bands + 1, device=x.device)
                for i in range(effective_n_freq_bands):
                    band_mask = (freqs >= freq_bands[i]) & (freqs < freq_bands[i + 1])
                    if band_mask.any():
                        band_power = power_spectrum[:, band_mask].mean(dim=-1)
                    else:
                        band_power = torch.zeros(power_spectrum.size(0), device=x.device)
                    band_powers.append(band_power)  # 修复缩进错误：这行应该在循环体内，不在else内
            
            freq_targets = torch.stack(band_powers, dim=-1)  # [B*R*E, n_freq_bands]
            freq_targets = freq_targets.view(B, R * E, effective_n_freq_bands)
        
        return time_targets, freq_targets


# 保留其他辅助函数和类（create_symmetric_masks, compute_frequency_metrics等）
def create_symmetric_masks(batch_size: int, 
                          seq_len: int, 
                          mask_ratio: float = 0.4,
                          mask_strategy: str = 'random',
                          device: torch.device = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """创建对称掩码"""
    if mask_strategy == 'random':
        mask1 = torch.rand(batch_size, seq_len, device=device) < mask_ratio
        mask2 = torch.rand(batch_size, seq_len, device=device) < mask_ratio
    elif mask_strategy == 'block':
        mask1 = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        mask2 = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        
        for b in range(batch_size):
            block_size = max(1, int(seq_len * mask_ratio))
            start_idx = torch.randint(0, seq_len - block_size + 1, (1,)).item()
            mask1[b, start_idx:start_idx + block_size] = True
            
            start_idx = torch.randint(0, seq_len - block_size + 1, (1,)).item()
            mask2[b, start_idx:start_idx + block_size] = True
    else:
        raise ValueError(f"Unknown mask strategy: {mask_strategy}")
    
    return mask1, mask2


def compute_frequency_metrics(original: torch.Tensor, 
                             reconstructed: torch.Tensor) -> Dict[str, float]:
    """计算频域指标"""
    # 使用float32进行FFT计算以避免cuFFT的half precision限制
    with torch.cuda.amp.autocast(enabled=False):
        original_float = original.float()  # 确保输入为 float32
        reconstructed_float = reconstructed.float()  # 确保输入为 float32
        
        # 计算功率谱密度
        orig_fft = torch.fft.rfft(original_float, dim=-1)
        recon_fft = torch.fft.rfft(reconstructed_float, dim=-1)
        
        orig_psd = torch.abs(orig_fft) ** 2
        recon_psd = torch.abs(recon_fft) ** 2
        
        # 计算频域MSE
        freq_mse = F.mse_loss(recon_psd, orig_psd)
    
    # 计算相位一致性
    orig_phase = torch.angle(orig_fft)
    recon_phase = torch.angle(recon_fft)
    phase_consistency = torch.cos(orig_phase - recon_phase).mean()
    
    return {
        'frequency_mse': freq_mse.item(),
        'phase_consistency': phase_consistency.item(),
    }


def create_frequency_masks(batch_size: int, n_freq_bands: int = 5, mask_ratio: float = 0.3, device: torch.device = 'cpu'):
    """创建频域掩码"""
    # 防护检查：确保n_freq_bands至少为1
    effective_n_freq_bands = max(1, n_freq_bands)
    n_masked_bands = max(1, int(effective_n_freq_bands * mask_ratio))
    mask = torch.zeros(batch_size, effective_n_freq_bands, device=device, dtype=torch.bool)
    
    for b in range(batch_size):
        masked_indices = torch.randperm(effective_n_freq_bands)[:n_masked_bands]
        mask[b, masked_indices] = True
    
    return mask


def compute_phase_consistency_loss(pred: torch.Tensor, target: torch.Tensor):
    """计算相位一致性损失"""
    # 使用float32进行FFT计算以避免cuFFT的half precision限制
    with torch.cuda.amp.autocast(enabled=False):
        pred_float = pred.float()  # 确保输入为 float32
        target_float = target.float()  # 确保输入为 float32
        
        pred_fft = torch.fft.rfft(pred_float, dim=-1)
        target_fft = torch.fft.rfft(target_float, dim=-1)
        
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # 计算相位差的余弦值，越接近1表示相位越一致
        phase_consistency = torch.cos(pred_phase - target_phase)
        
        # 损失为1减去相位一致性的平均值
        loss = 1.0 - phase_consistency.mean()
        
        return loss
    

class DualDomainLoss(nn.Module):
    """双域损失函数"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.time_weight = config.time_loss_weight
        self.freq_weight = config.freq_loss_weight
        self.moe_aux_coeff = config.moe_aux_loss_coeff
        
    def forward(self, time_pred1, time_pred2, time_targets, 
               freq_pred1, freq_pred2, freq_targets, 
                moe_aux_loss=None, padding_mask=None, **kwargs):
        
        # 在计算损失时，临时切换到 float32 以保证数值稳定性
        with torch.cuda.amp.autocast(enabled=False):
            # 将所有输入手动转换为 float32
            time_pred1_f32 = time_pred1.float()
            time_pred2_f32 = time_pred2.float()
            time_targets_f32 = time_targets.float()
            
            freq_pred1_f32 = freq_pred1.float()
            freq_pred2_f32 = freq_pred2.float()
            freq_targets_f32 = freq_targets.float()

            # 应用padding mask
            if padding_mask is not None:
                # padding_mask: [B, R, E] -> [B, R*E]
                mask = ~padding_mask.view(padding_mask.size(0), -1).unsqueeze(-1)  # [B, R*E, 1]
                
                # 对时域预测应用mask
                time_pred1_f32 = time_pred1_f32 * mask
                time_pred2_f32 = time_pred2_f32 * mask
                time_targets_f32 = time_targets_f32 * mask
                
                # 对频域预测应用mask
                freq_mask = mask.expand(-1, -1, freq_pred1_f32.size(-1))  # [B, R*E, n_freq_bands]
                freq_pred1_f32 = freq_pred1_f32 * freq_mask
                freq_pred2_f32 = freq_pred2_f32 * freq_mask
                freq_targets_f32 = freq_targets_f32 * freq_mask

            # 时域损失
            time_loss1 = F.mse_loss(time_pred1_f32, time_targets_f32)
            time_loss2 = F.mse_loss(time_pred2_f32, time_targets_f32)
            time_loss = (time_loss1 + time_loss2) / 2
                
            # 频域损失
            # 在计算MSE之前，对预测和目标应用log1p变换（使用float32确保数值稳定性）
            log_freq_pred1 = torch.log1p(F.relu(freq_pred1_f32)) # 使用ReLU确保输入非负
            log_freq_pred2 = torch.log1p(F.relu(freq_pred2_f32))
            log_freq_targets = torch.log1p(freq_targets_f32) # 目标已经是功率，保证非负

            freq_loss1 = F.mse_loss(log_freq_pred1, log_freq_targets)
            freq_loss2 = F.mse_loss(log_freq_pred2, log_freq_targets)
            freq_loss = (freq_loss1 + freq_loss2) / 2
            
            # 相位一致性损失
            phase_loss1 = compute_phase_consistency_loss(time_pred1_f32, time_targets_f32)
            phase_loss2 = compute_phase_consistency_loss(time_pred2_f32, time_targets_f32)
            phase_loss = (phase_loss1 + phase_loss2) / 2
            
            # 总损失
            total_loss = (self.time_weight * time_loss + 
                         self.freq_weight * freq_loss + 
                         0.1 * phase_loss)
            
            # 添加MoE辅助损失
            if moe_aux_loss is not None:
                total_loss = total_loss + self.moe_aux_coeff * moe_aux_loss.float()
        
        return {
            'loss': total_loss,
            'time_loss': time_loss.item(),
            'freq_loss': freq_loss.item(),
            'phase_loss': phase_loss.item(),
            'moe_loss': moe_aux_loss.item() if moe_aux_loss is not None else 0.0
        }


class EEGDataAugmentation(nn.Module):
    """EEG数据增强（保持不变）"""
    
    def __init__(
        self,
        p_noise: float = 0.5,
        noise_level: float = 0.05,
        p_channel_dropout: float = 0.3,
        dropout_ratio: float = 0.1,
        p_time_shift: float = 0.5,
        max_shift_ratio: float = 0.05
    ):
        super().__init__()
        self.p_noise = p_noise
        self.noise_level = noise_level
        self.p_channel_dropout = p_channel_dropout
        self.dropout_ratio = dropout_ratio
        self.p_time_shift = p_time_shift
        self.max_shift_ratio = max_shift_ratio
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入的EEG数据进行随机增强
        
        Args:
            x: 输入EEG数据 [B, R, E, T]
               
        Returns:
            增强后的EEG数据
        """
        if not self.training:
            return x
            
        # 随机选择要应用的增强方式
        augmentations = []
        
        if torch.rand(1) < self.p_noise:
            augmentations.append(self._add_gaussian_noise)
            
        if torch.rand(1) < self.p_channel_dropout:
            augmentations.append(self._channel_dropout)
            
        if torch.rand(1) < self.p_time_shift:
            augmentations.append(self._random_time_shift)
        
        # 依次应用选中的增强
        for aug_fn in augmentations:
            x = aug_fn(x)
            
        return x

    def _add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise_std = self.noise_level * torch.std(x, dim=-1, keepdim=True)
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    def _channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """随机失活通道（这里指电极）"""
        B, R, E, T = x.shape
        
        for b in range(B):
            for r in range(R):
                # 计算要失活的电极数量
                n_dropout = int(E * self.dropout_ratio)
                if n_dropout > 0:
                    # 随机选择要失活的电极
                    dropout_indices = torch.randperm(E)[:n_dropout]
                    x[b, r, dropout_indices] = 0
                    
        return x
    
    def _random_time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """随机时间偏移"""
        B, R, E, T = x.shape
        max_shift = int(T * self.max_shift_ratio)
        
        if max_shift > 0:
            for b in range(B):
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                if shift != 0:
                    x[b] = torch.roll(x[b], shift, dims=-1)
                    
        return x


def count_model_parameters(model, config: ModelConfig = None, verbose: bool = True):
    """
    计算模型参数量的详细统计信息
    
    Args:
        model: 要统计的模型实例
        config: 模型配置（可选，用于显示配置信息）
        verbose: 是否打印详细信息
        
    Returns:
        Dict: 包含各种参数统计的字典
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # 获取原始模型（去除DDP包装）
    raw_model = model.module if isinstance(model, DDP) else model
    
    # 基本参数统计
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # 按主要组件统计参数量
    component_stats = {}
    for name, module in raw_model.named_children():
        if hasattr(module, 'parameters'):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_stats[name] = params
    
    # 按模块类型统计参数量
    module_type_stats = {}
    for name, module in raw_model.named_modules():
        module_type = type(module).__name__
        if module_type not in module_type_stats:
            module_type_stats[module_type] = {'count': 0, 'params': 0}
        
        # 只统计直接属于该模块的参数，避免重复计算
        direct_params = sum(p.numel() for p in module.parameters(recurse=False))
        if direct_params > 0:
            module_type_stats[module_type]['count'] += 1
            module_type_stats[module_type]['params'] += direct_params
    
    # 移除参数为0的模块类型
    module_type_stats = {k: v for k, v in module_type_stats.items() if v['params'] > 0}
    
    # 内存占用估算（MB）
    fp32_memory = total_params * 4 / 1024 / 1024  # 4 bytes per parameter
    fp16_memory = total_params * 2 / 1024 / 1024  # 2 bytes per parameter
    
    # 梯度内存占用（训练时）
    gradient_memory = trainable_params * 4 / 1024 / 1024  # FP32 gradients
    
    # Adam优化器状态内存占用（m和v状态）
    adam_memory = trainable_params * 8 / 1024 / 1024  # 2 states * 4 bytes each
    
    # 总训练内存估算（模型 + 梯度 + 优化器状态）
    total_training_memory_fp32 = fp32_memory + gradient_memory + adam_memory
    total_training_memory_fp16 = fp16_memory + gradient_memory + adam_memory
    
    # 构建结果字典
    stats = {
        'basic': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'trainable_ratio': trainable_params / total_params * 100 if total_params > 0 else 0
        },
        'memory': {
            'model_fp32_mb': fp32_memory,
            'model_fp16_mb': fp16_memory,
            'gradient_mb': gradient_memory,
            'adam_states_mb': adam_memory,
            'total_training_fp32_mb': total_training_memory_fp32,
            'total_training_fp16_mb': total_training_memory_fp16
        },
        'components': component_stats,
        'module_types': module_type_stats,
        'model_class': type(raw_model).__name__
    }
    
    if verbose:
        print("\n" + "="*80)
        print(f"{'模型参数统计报告':^80}")
        print("="*80)
        
        # 基本信息
        print(f"\n📊 {'基本参数统计':^70}")
        print("-" * 75)
        print(f"{'模型类型:':<20} {stats['model_class']}")
        print(f"{'总参数量:':<20} {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"{'可训练参数:':<20} {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"{'冻结参数:':<20} {frozen_params:,} ({frozen_params/1e6:.2f}M)")
        print(f"{'可训练比例:':<20} {stats['basic']['trainable_ratio']:.1f}%")
        
        # 内存占用
        print(f"\n💾 {'内存占用估算':^70}")
        print("-" * 75)
        print(f"{'模型参数(FP32):':<20} {fp32_memory:.1f} MB")
        print(f"{'模型参数(FP16):':<20} {fp16_memory:.1f} MB")
        print(f"{'梯度(FP32):':<20} {gradient_memory:.1f} MB")
        print(f"{'Adam状态:':<20} {adam_memory:.1f} MB")
        print(f"{'训练总内存(FP32):':<20} {total_training_memory_fp32:.1f} MB ({total_training_memory_fp32/1024:.2f} GB)")
        print(f"{'训练总内存(FP16):':<20} {total_training_memory_fp16:.1f} MB ({total_training_memory_fp16/1024:.2f} GB)")
        
        # 主要组件分布
        if component_stats:
            print(f"\n🏗️  {'主要组件参数分布':^70}")
            print("-" * 75)
            sorted_components = sorted(component_stats.items(), key=lambda x: x[1], reverse=True)
            for name, params in sorted_components:
                percentage = params / total_params * 100
                print(f"{name:<30} {params:>15,} ({percentage:>5.1f}%)")
        
        # 模块类型统计
        if module_type_stats:
            print(f"\n🔧 {'模块类型统计':^70}")
            print("-" * 75)
            print(f"{'模块类型':<25} {'实例数':<8} {'参数量':<15} {'占比'}")
            print("-" * 75)
            sorted_modules = sorted(module_type_stats.items(), key=lambda x: x[1]['params'], reverse=True)
            for module_type, info in sorted_modules:
                percentage = info['params'] / total_params * 100
                print(f"{module_type:<25} {info['count']:<8} {info['params']:>12,} ({percentage:>5.1f}%)")
        
        # 配置信息（如果提供）
        if config:
            print(f"\n⚙️  {'模型配置信息':^70}")
            print("-" * 75)
            print(f"{'嵌入维度:':<20} {config.embed_dim}")
            print(f"{'注意力头数:':<20} {config.num_heads}")
            print(f"{'Transformer层数:':<20} {config.depth}")
            print(f"{'MLP比例:':<20} {config.mlp_ratio}")
            if hasattr(config, 'use_moe') and config.use_moe:
                print(f"{'使用MoE:':<20} Yes (专家数: {config.num_experts}, Top-K: {config.top_k_experts})")
            else:
                print(f"{'使用MoE:':<20} No")
            print(f"{'序列长度:':<20} {config.num_regions} × {config.max_electrodes_per_region} = {config.num_regions * config.max_electrodes_per_region}")
            print(f"{'输入维度:':<20} [{config.num_regions}, {config.max_electrodes_per_region}, {config.sequence_length}]")

    
    return stats


# 使用示例
if __name__ == "__main__":
    # 创建配置和模型
    config = ModelConfig()
    model = DualDomainTransformerMEM(config)
    
    # 计算参数量
    stats = count_model_parameters(model, config, verbose=True)

