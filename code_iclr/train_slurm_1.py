# -*- coding: utf-8 -*-

import os
import torch
# 启用 TensorFloat32 核心以提升性能
#torch.set_float32_matmul_precision('high') 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Tuple, Optional, List
import time
import warnings
import gc
import math

# 屏蔽所有torchvision相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
# 屏蔽学习率调度器警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
# 屏蔽其他常见警告
warnings.filterwarnings("ignore", message=".*skipping the first value of the learning rate schedule.*")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

# 导入模型和数据加载器
from model_slurm_1 import (DualDomainTransformerMEM, ModelConfig, 
                  create_symmetric_masks, create_frequency_masks, 
                  compute_frequency_metrics, compute_phase_consistency_loss,
                  DualDomainLoss, EEGDataAugmentation)
from load_slurm_1 import EEGBrainRegionDataset, create_data_loader

# ========================= DDP 相关函数 =========================
def setup_ddp():
    """初始化DDP环境，使用Slurm提供的环境变量"""
    # 从Slurm环境变量获取分布式信息
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    
    # 获取master节点信息
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    # 设置环境变量（以防未设置）
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # 绑定GPU
    torch.cuda.set_device(local_rank)
    
    # 初始化分布式进程组
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )
    
    # 同步所有进程
    dist.barrier()
    
    return rank, local_rank, world_size

def cleanup_ddp():
    """清理DDP环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

def print_rank_0(message, end='\n'):
    """只在主进程中打印消息"""
    if is_main_process():
        print(message, end=end)

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """在所有进程间聚合张量并返回平均值"""
    if not dist.is_initialized():
        return tensor
    
    # 确保tensor在正确的设备上
    if not tensor.is_cuda:
        tensor = tensor.cuda()
    
    # 克隆tensor以避免修改原始值
    reduced_tensor = tensor.clone()
    
    # 执行all_reduce操作
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.AVG)
    
    return reduced_tensor

def aggregate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """聚合所有进程的指标"""
    if not dist.is_initialized():
        return metrics
    
    aggregated_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # 将标量转换为tensor进行聚合
            tensor_value = torch.tensor(value, dtype=torch.float32, device=torch.cuda.current_device())
            reduced_value = reduce_tensor(tensor_value)
            aggregated_metrics[key] = reduced_value.item()
        else:
            aggregated_metrics[key] = value
    
    return aggregated_metrics

# ========================= DualDomainTrainer 类 =========================

class DualDomainTrainer:
    """封装时域和频域多任务学习训练逻辑 - DDP版本"""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, config: ModelConfig, device: torch.device, rank: int = 0):
        self.config = config
        self.device = device
        self.rank = rank
        self.is_ddp = dist.is_initialized()
        
        # 接受外部传入的模型和优化器
        self.model = model
        self.optimizer = optimizer
        
        # 在DDP环境中，有效批次大小是所有进程的总和
        if self.is_ddp:
            self.effective_batch_size = config.batch_size * dist.get_world_size()
            print_rank_0(f"DDP训练，世界大小: {dist.get_world_size()}, 有效批次大小: {self.effective_batch_size}")
        else:
            self.effective_batch_size = config.batch_size
            print_rank_0(f"单GPU训练，批次大小: {self.effective_batch_size}")
        
        # 初始化数据增强模块
        self.data_augmentation = EEGDataAugmentation(
            p_noise=0.5,                # 50%概率应用噪声
            noise_level=0.05,           # 噪声强度为信号标准差的5%
            p_channel_dropout=0.3,      # 30%概率应用通道失活
            dropout_ratio=0.1,          # 随机失活10%的电极
            p_time_shift=0.4,           # 40%概率应用时间偏移
            max_shift_ratio=0.05        # 最大偏移为时间轴5%
        ).to(device)
        
        # 学习率调度器将在设置训练数据加载器后初始化
        self.lr_scheduler = None
        self.steps_per_epoch = None  # 将在setup_training中设置
        
        # 多任务损失函数 - 支持MoE辅助损失
        self.loss_fn = DualDomainLoss(config)
        self.loss_scale = 0.1  # 添加损失缩放因子
        
        # 混合精度训练
        self.use_amp = config.use_amp
        self.scaler = GradScaler(init_scale=2**16) if config.use_amp else None  # 降低初始缩放因子
        
        # 训练状态
        self.global_step = 0
        
        # 时间管理 - 统一初始化
        self.training_start_time = None  # 将在train方法开始时设置
        self.last_periodic_save_time = None  # 周期性保存计时器
        
        # 梯度累积步数 - 动态计算
        target_global_batch_size = 128  # 目标全局批次大小
        current_global_batch_size = config.batch_size * (dist.get_world_size() if self.is_ddp else 1)
        self.gradient_accumulation_steps = max(1, target_global_batch_size // current_global_batch_size)
        
        print_rank_0(f"动态梯度累积设置:")
        print_rank_0(f"  目标全局批次大小: {target_global_batch_size}")
        print_rank_0(f"  当前全局批次大小: {current_global_batch_size}")
        print_rank_0(f"  梯度累积步数: {self.gradient_accumulation_steps}")
        print_rank_0(f"  实际有效全局批次大小: {current_global_batch_size * self.gradient_accumulation_steps}")
        
        # 计算并存储模型参数统计信息
        self.param_stats = self._calculate_model_parameters()
        
        # 训练指标
        self.metrics = {
            'train_loss': [],
            'time_loss': [],
            'freq_loss': [],
            'phase_loss': []
        }
        
        # 最佳损失跟踪
        self.best_loss = float('inf')
        
        # 创建本次运行的文件夹
        run_name = time.strftime('%Y%m%d_%H%M%S')
        self.run_checkpoint_dir = f"checkpoint/{run_name}"
        os.makedirs(self.run_checkpoint_dir, exist_ok=True)
        
        # 日志记录器
        self.tensorboard_writer = None
        self.log_dir = None
    
    def _calculate_model_parameters(self):
        """计算模型参数量的详细统计"""
        # 获取原始模型（去除DDP包装）
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # 按主要组件统计参数量
        component_params = {}
        
        # 统计各个主要组件的参数量
        for name, module in raw_model.named_children():
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_params[name] = params
        
        # 按模块类型统计
        module_type_params = {}
        for name, module in raw_model.named_modules():
            module_type = type(module).__name__
            if module_type not in module_type_params:
                module_type_params[module_type] = 0
            # 只统计直接属于该模块的参数，避免重复计算
            direct_params = sum(p.numel() for p in module.parameters(recurse=False))
            module_type_params[module_type] += direct_params
        
        # 移除参数为0的模块类型
        module_type_params = {k: v for k, v in module_type_params.items() if v > 0}
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'components': component_params,
            'module_types': module_type_params,
            'memory_fp32_mb': total_params * 4 / 1024 / 1024,
            'memory_fp16_mb': total_params * 2 / 1024 / 1024
        }
        
    def setup_training(self, train_loader, num_epochs=None):
        """设置训练数据加载器和相关参数"""
        self.steps_per_epoch = len(train_loader)
        self.num_epochs = num_epochs or self.config.warmup_epochs
        
        # 现在可以正确初始化学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.lr,
            epochs=self.num_epochs,  # 使用实际的训练epochs
            steps_per_epoch=self.steps_per_epoch,  # 使用实际的每个epoch的步数
            pct_start=0.5,  # 增加预热阶段比例
            anneal_strategy='cos',
            div_factor=50.0,  # 增加初始学习率降低程度
            final_div_factor=2000.0  # 增加最终学习率降低程度
        )
        
        print_rank_0(f"学习率调度器已初始化，总训练epochs: {self.num_epochs}, 每个epoch的步数: {self.steps_per_epoch}")
        print_rank_0(f"总训练步数: {self.num_epochs * self.steps_per_epoch}")
    
    def setup_logging(self, log_dir='./logs', record_graph=False):
        """设置TensorBoard日志记录
        
        Args:
            log_dir: 日志保存目录
            record_graph: 是否记录模型计算图，设置为False可避免TracerWarning和维度不匹配问题
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            import os
            
            # 创建日志目录
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
            
            # 创建更具描述性的运行名称
            run_name = (f"{time.strftime('%Y%m%d-%H%M%S')}_"
                       f"emb{self.config.embed_dim}_"
                       f"depth{self.config.depth}_"
                       f"bs{self.config.batch_size}")
            
            # 创建日志写入器
            log_path = f"{log_dir}/{run_name}"
            self.tensorboard_writer = SummaryWriter(log_path)
            
            print_rank_0(f"TensorBoard日志将保存到 {log_path}")
            
            # 记录模型图（可选）
            if record_graph:
                try:
                    # 创建与模型期望形状匹配的dummy_input
                    # 首先获取模型期望的输入形状
                    batch_size = 2  # 使用较小的批次大小避免内存问题
                    in_channels = self.config.in_chans
                    
                    # 使用模型中实际的电极数量作为H维度
                    if hasattr(self.config, 'electrode_names') and self.config.electrode_names:
                        height = len(self.config.electrode_names)
                    else:
                        height = 22  # 默认电极数量
                        
                    # 使用较小的时间维度以减少内存使用
                    width = 160
                    
                    print_rank_0(f"创建示例输入，形状: [{batch_size}, {in_channels}, {height}, {width}]")
                    
                    # 创建dummy_input
                    dummy_input = torch.randn(batch_size, in_channels, height, width, device=self.device)
                    
                    # 获取原始模型用于图记录
                    raw_model = self.model.module if isinstance(self.model, DDP) else self.model
                    
                    # 首先验证输入形状是否正确
                    with torch.no_grad():
                        try:
                            time_features, freq_features, raw_features = raw_model.dual_proj(dummy_input)
                            print_rank_0(f"输入验证成功: 时域特征: {time_features.shape}, 频域特征: {freq_features.shape}, 原始信号特征: {raw_features.shape}")
                            
                            # 再尝试完整的前向传播
                            _ = raw_model(dummy_input)
                            print_rank_0("完整前向传播验证成功，记录模型图...")
                            
                            # 记录模型图（使用原始模型）
                            self.tensorboard_writer.add_graph(raw_model, dummy_input)
                            print_rank_0("模型图已成功记录到TensorBoard")
                        except Exception as e:
                            print_rank_0(f"输入验证失败: {str(e)}")
                            print_rank_0("尝试不记录模型图而继续训练...")
                except Exception as e:
                    print_rank_0(f"无法记录模型图: {str(e)}")
                    print_rank_0("跳过模型图记录，继续训练...")
            else:
                print_rank_0("已禁用模型图记录以避免TracerWarning和维度不匹配问题")
            
            # 记录超参数和模型参数统计
            try:
                hparams = {k: v for k, v in self.config.__dict__.items() 
                          if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
                # 添加模型参数统计到超参数
                hparams.update({
                    'model/total_params': self.param_stats['total'],
                    'model/trainable_params': self.param_stats['trainable'],
                    'model/memory_fp32_mb': self.param_stats['memory_fp32_mb'],
                    'model/memory_fp16_mb': self.param_stats['memory_fp16_mb']
                })
                metric_dict = {'hparam/val_loss': 0}  # 使用更有意义的指标名称
                self.tensorboard_writer.add_hparams(hparams, metric_dict)
                
                # 记录模型参数统计信息
                self.tensorboard_writer.add_scalar('model/total_parameters', self.param_stats['total'], 0)
                self.tensorboard_writer.add_scalar('model/trainable_parameters', self.param_stats['trainable'], 0)
                self.tensorboard_writer.add_scalar('model/memory_fp32_mb', self.param_stats['memory_fp32_mb'], 0)
                self.tensorboard_writer.add_scalar('model/memory_fp16_mb', self.param_stats['memory_fp16_mb'], 0)
                
                print_rank_0("超参数和模型参数统计已记录到TensorBoard")
            except Exception as e:
                print_rank_0(f"记录超参数失败: {str(e)}")
            
            return True
        except ImportError:
            print_rank_0("警告: 未找到TensorBoard，将不会记录训练日志")
            return False
        except Exception as e:
            print_rank_0(f"设置TensorBoard日志失败: {str(e)}")
            return False
    
    def log_metrics(self, metrics, step, prefix='train'):
        """记录指标到TensorBoard"""
        if self.tensorboard_writer is None:
            return
        
        # 记录标量值
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.tensorboard_writer.add_scalar(f"{prefix}/{key}", value, step)
            elif isinstance(value, dict):
                # 处理嵌套字典
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (int, float)):
                        self.tensorboard_writer.add_scalar(
                            f"{prefix}/{key}/{nested_key}",
                            nested_value,
                            step
                        )
        
        # 记录学习率
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.tensorboard_writer.add_scalar(f"{prefix}/learning_rate", current_lr, step)
    
    def _validate_input_format(self, x: torch.Tensor, padding_mask: torch.Tensor) -> None:
        """验证输入数据格式是否正确
        
        Args:
            x: 输入数据张量 [B, R, E, T]
            padding_mask: 填充掩码 [B, R, E]
        """
        # 验证输入数据维度
        if x.dim() != 4:
            raise ValueError(f"期望4D输入张量 [B, R, E, T]，得到 {x.dim()}D: {x.shape}")
        
        # 验证padding mask维度
        if padding_mask.dim() != 3:
            raise ValueError(f"期望3D填充掩码 [B, R, E]，得到 {padding_mask.dim()}D: {padding_mask.shape}")
        
        # 验证维度匹配
        B, R, E, T = x.shape
        if padding_mask.shape != (B, R, E):
            raise ValueError(f"数据形状 {x.shape} 与掩码形状 {padding_mask.shape} 不匹配")
        
        # 只在第一步打印维度信息
        if self.global_step == 0:
            print_rank_0(f"\n=== 输入数据格式验证 ===")
            print_rank_0(f"数据形状: {x.shape} [批次, 脑区, 电极, 时间]")
            print_rank_0(f"掩码形状: {padding_mask.shape} [批次, 脑区, 电极]")
            print_rank_0(f"脑区数量: {R}")
            print_rank_0(f"每个脑区最大电极数: {E}")
            print_rank_0(f"时间序列长度: {T}")
            
            # 统计真实电极数量
            for r in range(R):
                real_electrodes = (~padding_mask[0, r]).sum().item()
                print_rank_0(f"  脑区 {r}: {real_electrodes} 个真实电极")
            print_rank_0("=" * 30)
    
    def _run_one_step(self, x: torch.Tensor, padding_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """执行模型前向传播和损失计算的核心流程
        
        参数:
            x: 输入数据 [B, R, E, T]
            padding_mask: 填充遮罩 [B, R, E]，True表示填充位置
            
        返回:
            结果字典，包含以下键:
            - time_pred1, time_pred2: 时域预测
            - freq_pred1, freq_pred2: 频域预测
            - time_targets, freq_targets: 目标特征
            - loss_dict: 损失字典
            - loss: 总损失(已乘以loss_scale)
            - region_losses: 脑区级别损失字典
            - moe_aux_loss: MoE辅助损失 (如果使用MoE)
        """
        # 获取原始模型（去除DDP包装）
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        
        # 验证输入格式
        self._validate_input_format(x, padding_mask)
        
        # 添加GPU同步点，确保所有GPU操作完成
        torch.cuda.synchronize()
        
        # 获取数据维度
        B, R, E, T = x.shape
        seq_len = R * E  # 序列长度现在是脑区数×电极数
        
        # 只在第一个批次打印维度信息（已在_validate_input_format中处理）
        
        # 创建时域掩码 - 现在基于序列长度R*E
        time_mask1, _ = create_symmetric_masks(
            B, seq_len, 
            mask_ratio=self.config.mask_ratio,
            mask_strategy=self.config.mask_strategy,
            device=self.device
        )
        
        # 创建频域掩码
        freq_mask1 = create_frequency_masks(
            B, self.config.n_freq_bands,
            mask_ratio=self.config.freq_mask_ratio,
            device=self.device
        )
        
        # 前向计算 - 获取时域和频域预测，以及MoE辅助损失
        time_pred1, time_pred2, freq_pred1, freq_pred2, moe_aux_loss = self.model(
            x, time_mask=time_mask1, freq_mask=freq_mask1, padding_mask=padding_mask
        )
        
        # 目标特征
        time_targets, freq_targets = raw_model.get_targets(x, padding_mask=padding_mask)
        
        # 计算多任务损失，包括MoE辅助损失
        loss_dict = self.loss_fn(
            time_pred1, time_pred2, time_targets,
            freq_pred1, freq_pred2, freq_targets,
            moe_aux_loss=moe_aux_loss,
            padding_mask=padding_mask
        )
        
        # 总损失 - 确保在多GPU环境下损失是标量
        loss = loss_dict['loss']
        if loss.dim() > 0:  # 如果损失不是标量（多GPU情况）
            loss = loss.mean()  # 取平均值
        loss = loss * self.loss_scale
        
        # 计算脑区级别的损失
        region_losses = self._compute_region_losses(time_pred1, time_targets, freq_pred1, freq_targets)
        
        # 返回所有结果
        return {
            'time_pred1': time_pred1,
            'time_pred2': time_pred2,
            'freq_pred1': freq_pred1,
            'freq_pred2': freq_pred2,
            'time_targets': time_targets,
            'freq_targets': freq_targets,
            'loss_dict': loss_dict,
            'loss': loss,
            'region_losses': region_losses,
            'moe_aux_loss': moe_aux_loss
        }
    
    def train_step(self, batch) -> Dict[str, float]:
        """执行一步多任务训练，包括时域和频域任务"""
        try:
            self.model.train()
            
            # 处理重构后的数据格式：(data_batch, mask_batch)
            if isinstance(batch, tuple) and len(batch) == 2:
                # 新的数据格式：(grouped_data_tensor, padding_mask_tensor)
                x_data, padding_mask = batch
                # x_data: [B, R, E, T] - 已按脑区分组
                # padding_mask: [B, R, E] - True表示填充位置
            elif isinstance(batch, (tuple, list)) and len(batch) > 0:
                # 旧格式兼容性
                x_data = batch[0]
                # 为旧格式创建默认padding mask
                B, R, E = x_data.shape[0], x_data.shape[1], x_data.shape[2]
                padding_mask = torch.zeros(B, R, E, dtype=torch.bool)
            elif isinstance(batch, torch.Tensor):
                # 直接张量输入
                x_data = batch
                B, R, E = x_data.shape[0], x_data.shape[1], x_data.shape[2]
                padding_mask = torch.zeros(B, R, E, dtype=torch.bool)
            else:
                raise TypeError(f"不支持的批次数据格式：{type(batch)}")
            
            # 确保数据和遮罩在正确的设备上
            x = x_data.to(self.device)  # [B, R, E, T]
            padding_mask = padding_mask.to(self.device)  # [B, R, E]
            
            # 应用数据增强（只在训练模式下）
            # 设置增强模块为训练模式
            self.data_augmentation.train()
            x = self.data_augmentation(x)
            
            # 启用混合精度训练
            if self.use_amp:
                # 获取数据维度
                B, R, E, T = x.shape
                seq_len = R * E  # 序列长度现在是脑区数×电极数
                
                # 创建时域掩码 - 现在基于序列长度R*E
                time_mask1, _ = create_symmetric_masks(
                    B, seq_len, 
                    mask_ratio=self.config.mask_ratio,
                    mask_strategy=self.config.mask_strategy,
                    device=self.device
                )
                
                # 创建频域掩码
                freq_mask1 = create_frequency_masks(
                    B, self.config.n_freq_bands,
                    mask_ratio=self.config.freq_mask_ratio,
                    device=self.device
                )
                
                with autocast():
                    # 执行模型前向传播（不包括损失计算）
                    time_pred1, time_pred2, freq_pred1, freq_pred2, moe_aux_loss = self.model(
                        x, time_mask=time_mask1, freq_mask=freq_mask1, padding_mask=padding_mask
                    )
                    
                    # 目标特征
                    raw_model = self.model.module if isinstance(self.model, DDP) else self.model
                    time_targets, freq_targets = raw_model.get_targets(x, padding_mask=padding_mask)
                
                # 在autocast上下文外计算损失，确保数值稳定性
                loss_dict = self.loss_fn(
                    time_pred1, time_pred2, time_targets,
                    freq_pred1, freq_pred2, freq_targets,
                    moe_aux_loss=moe_aux_loss,
                    padding_mask=padding_mask
                )
                
                loss = loss_dict['loss']
                if loss.dim() > 0:  # 如果损失不是标量（多GPU情况）
                    loss = loss.mean()  # 取平均值
                loss = loss * self.loss_scale
                
                # 计算脑区级别的损失
                region_losses = self._compute_region_losses(time_pred1, time_targets, freq_pred1, freq_targets)
                
                # 梯度缩放和反向传播
                scaled_loss = self.scaler.scale(loss / self.gradient_accumulation_steps)
                scaled_loss.backward()
                
                # 梯度累积
                if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.config.clip_grad > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                        
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # 添加GPU同步点
                    torch.cuda.synchronize()
                    
                    # 更新学习率 - 移到optimizer.step()后面，添加步数检查
                    self._safe_lr_step()
            else:
                # 不使用混合精度的标准训练
                # 执行核心前向传播和损失计算
                result = self._run_one_step(x, padding_mask)
                loss = result['loss']
                # 确保在多GPU环境下损失是标量
                if loss.dim() > 0:  # 如果损失不是标量（多GPU情况）
                    loss = loss.mean()  # 取平均值
                loss_dict = result['loss_dict']
                region_losses = result['region_losses']
                
                # 反向传播
                loss_grad = loss / self.gradient_accumulation_steps
                loss_grad.backward()
                
                # 梯度累积
                if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.config.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                        
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # 添加GPU同步点
                    torch.cuda.synchronize()
                    
                    # 更新学习率 - 移到optimizer.step()后面，添加步数检查
                    self._safe_lr_step()
            
            # 更新全局步数
            self.global_step += 1
            
            # 返回损失指标
            metrics = {
                'loss': loss.item() * self.gradient_accumulation_steps,
                'time_loss': loss_dict['time_loss'],
                'freq_loss': loss_dict['freq_loss'],
                'phase_loss': loss_dict['phase_loss'],
                'moe_loss': loss_dict.get('moe_loss', 0.0)
            }
            
            # 添加脑区级别的损失
            for region_name, region_loss in region_losses.items():
                metrics[region_name] = region_loss
            
            # 在DDP环境中聚合指标
            aggregated_metrics = aggregate_metrics(metrics)
            
            # 记录指标到TensorBoard（只在主进程中）
            if self.rank == 0 and self.global_step % 10 == 0:  # 每10步记录一次，避免日志过大
                self.log_metrics(aggregated_metrics, self.global_step, prefix='train')
            
            return aggregated_metrics
            
        except RuntimeError as e:
            if "NCCL" in str(e) or "corrupted" in str(e):
                print_rank_0(f"捕获到NCCL错误: {str(e)}")
                # 清理GPU内存
                torch.cuda.empty_cache()
                gc.collect()
                
                # 重置优化器状态
                self.optimizer.zero_grad()
                if self.use_amp and self.scaler is not None:
                    # 重置scaler状态而不是更新
                    try:
                        # 重新初始化scaler以清除损坏的状态
                        self.scaler = GradScaler(init_scale=2**16)
                        print_rank_0("scaler已重新初始化")
                    except Exception as scaler_error:
                        print_rank_0(f"警告: scaler重新初始化失败: {str(scaler_error)}")
                
                # 添加GPU同步
                torch.cuda.synchronize()
                
                # 返回默认指标，避免训练中断
                return {
                    'loss': 1.0,
                    'time_loss': 0.5,
                    'freq_loss': 0.5,
                    'phase_loss': 0.0,
                    'moe_loss': 0.0
                }
            else:
                # 其他错误直接抛出
                raise e
    
    def _safe_lr_step(self):
        """安全地执行学习率调度器步进，防止超过总步数"""
        try:
            # 检查OneCycleLR调度器的步数是否超过总步数
            if hasattr(self.lr_scheduler, 'total_steps') and hasattr(self.lr_scheduler, 'last_epoch'):
                if self.lr_scheduler.last_epoch >= self.lr_scheduler.total_steps:
                    # 如果已经达到总步数，切换到简单的调度器
                    self._switch_to_simple_scheduler()
                    return
            
            # 正常步进
            self.lr_scheduler.step()
        except ValueError as e:
            if "Tried to step" in str(e) and "total steps" in str(e):
                # 如果步数超过总步数，切换到简单的调度器
                print_rank_0(f"警告: 学习率调度器步数超限，切换到简单调度器 - {str(e)}")
                self._switch_to_simple_scheduler()
                return
            else:
                # 其他错误重新抛出
                raise e
        except Exception as e:
            print_rank_0(f"学习率调度器步进时发生错误: {str(e)}")
            # 可以选择继续训练或者抛出错误
            pass
    
    def _switch_to_simple_scheduler(self):
        """切换到简单的学习率调度器"""
        if not hasattr(self, '_switched_to_simple'):
            print_rank_0("正在切换到简单的学习率调度器(ExponentialLR)...")
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.99  # 每步乘以0.99，缓慢衰减
            )
            # 设置当前学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            self._switched_to_simple = True
            print_rank_0(f"已切换到简单调度器，当前学习率: {current_lr:.8f}")
        
        # 执行简单调度器的步进
        self.lr_scheduler.step()
    
    def _compute_region_losses(self, time_pred, time_target, freq_pred=None, freq_target=None):
        """计算不同脑区的重建损失
        
        Args:
            time_pred: 时域预测 [B, R*E, T] 或 [B, R*E, D]
            time_target: 时域目标 [B, R*E, T] 或 [B, R*E, D]  
            freq_pred: 频域预测 (可选) [B, R*E, n_freq_bands]
            freq_target: 频域目标 (可选) [B, R*E, n_freq_bands]
            
        Returns:
            包含不同脑区损失的字典，键名格式为'region_time_loss'或'region_freq_loss'
        """
        region_losses = {}
        
        # 检查预测的序列长度
        seq_len = time_pred.size(1)  # 序列长度 R*E
        
        # 日志记录以帮助调试
        if self.global_step == 0:
            print_rank_0(f"序列长度 (R*E): {seq_len}")
            print_rank_0(f"时域预测形状: {time_pred.shape}, 目标形状: {time_target.shape}")
            if freq_pred is not None:
                print_rank_0(f"频域预测形状: {freq_pred.shape}, 目标形状: {freq_target.shape}")
        
        # 新的数据格式下，序列维度是 R*E（脑区数×电极数）
        # 脑区数固定为5，电极数为24
        num_regions = self.config.num_regions  # 5
        max_electrodes_per_region = self.config.max_electrodes_per_region  # 24
        
        # 确保序列长度匹配预期
        expected_seq_len = num_regions * max_electrodes_per_region
        if seq_len != expected_seq_len:
            print_rank_0(f"警告: 序列长度 {seq_len} 不匹配预期 {expected_seq_len}")
        
        # 遍历脑区并计算每个脑区的损失
        for region_idx, region_name in enumerate(['frontal', 'central', 'parietal', 'temporal', 'occipital']):
            # 计算当前脑区在序列中的位置范围
            start_idx = region_idx * max_electrodes_per_region
            end_idx = (region_idx + 1) * max_electrodes_per_region
            
            # 提取当前脑区的预测和目标
            region_time_pred = time_pred[:, start_idx:end_idx]  # [B, E, T] or [B, E, D]
            region_time_target = time_target[:, start_idx:end_idx]  # [B, E, T] or [B, E, D]
            
            # 计算时域损失
            time_region_loss = F.mse_loss(region_time_pred, region_time_target)
            region_losses[f"{region_name}_time_loss"] = time_region_loss.item()
            
            # 如果有频域预测，也计算频域损失
            if freq_pred is not None and freq_target is not None:
                region_freq_pred = freq_pred[:, start_idx:end_idx]  # [B, E, n_freq_bands]
                region_freq_target = freq_target[:, start_idx:end_idx]  # [B, E, n_freq_bands]
                
                freq_region_loss = F.mse_loss(region_freq_pred, region_freq_target)
                region_losses[f"{region_name}_freq_loss"] = freq_region_loss.item()
        
        return region_losses
    
    # 验证方法已移除 - 将在单独的验证脚本中实现
    
    def train(self, train_loader: torch.utils.data.DataLoader,
              sampler: Optional[DistributedSampler] = None,
              num_epochs: int = 10,
              log_dir: str = './logs') -> Dict:
        
        # =================================================================
        # === 步骤 1: 定义时间控制的检查点参数 ===
        # =================================================================
        checkpoint_interval_hours = 4  # 您可以轻松修改这里，例如改为2、6或8小时
        checkpoint_interval_seconds = checkpoint_interval_hours * 3600
        
        # 为了避免在每个step都检查时间带来的开销，我们每隔100步检查一次
        check_time_every_n_steps = 100 
        # =================================================================
        
        # 设置训练数据加载器和学习率调度器
        try:
            self.setup_training(train_loader, num_epochs)
            print_rank_0("学习率调度器设置成功")
        except Exception as e:
            print_rank_0(f"设置学习率调度器失败: {str(e)}")
            print_rank_0("使用默认学习率调度器")
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
            self.steps_per_epoch = len(train_loader)
        if is_main_process():
            self.setup_logging(log_dir, record_graph=False)
        print_rank_0("\n" + "="*50)
        print_rank_0(f"{'DualDomain Neural Transformer MEM 模型训练开始':^50}")
        print_rank_0("="*50)
        print_rank_0(f"{'本次运行文件夹':^50}")
        print_rank_0("-"*50)
        print_rank_0(f"{'运行文件夹:':<15} {self.run_checkpoint_dir}")
        print_rank_0(f"{'最佳模型:':<15} {self.run_checkpoint_dir}/best_model.pt")
        print_rank_0(f"{'最后模型:':<15} {self.run_checkpoint_dir}/last_model.pt")
        print_rank_0("-"*50)
        print_rank_0(f"{'模型参数统计':^50}")
        print_rank_0("-"*50)
        print_rank_0(f"{'总参数量:':<15} {self.param_stats['total']:,}")
        print_rank_0(f"{'可训练参数:':<15} {self.param_stats['trainable']:,}")
        print_rank_0(f"{'冻结参数:':<15} {self.param_stats['frozen']:,}")
        print_rank_0(f"{'内存占用(FP32):':<15} {self.param_stats['memory_fp32_mb']:.1f} MB")
        print_rank_0(f"{'内存占用(FP16):':<15} {self.param_stats['memory_fp16_mb']:.1f} MB")
        print_rank_0("")
        print_rank_0(f"{'主要组件参数分布':^50}")
        print_rank_0("-"*50)
        if is_main_process():
            for name, params in sorted(self.param_stats['components'].items(), key=lambda x: x[1], reverse=True):
                percentage = params / self.param_stats['total'] * 100
                print_rank_0(f"{name:<20} {params:>12,} ({percentage:>5.1f}%)")
        print_rank_0("")
        print_rank_0(f"{'训练配置信息':^50}")
        print_rank_0("-"*50)
        print_rank_0(f"{'训练周期:':<15} {num_epochs}")
        print_rank_0(f"{'批次大小:':<15} {train_loader.batch_size}")
        print_rank_0(f"{'有效批次大小:':<15} {self.effective_batch_size}")
        print_rank_0(f"{'设备:':<15} {self.device}")
        if self.is_ddp:
            print_rank_0(f"{'DDP世界大小:':<15} {dist.get_world_size()}")
            print_rank_0(f"{'DDP模式:':<15} 启用")
        else:
            print_rank_0(f"{'DDP模式:':<15} 禁用")
        print_rank_0(f"{'学习率:':<15} {self.config.lr}")
        print_rank_0(f"{'时域损失权重:':<15} {self.config.time_loss_weight}")
        print_rank_0(f"{'频域损失权重:':<15} {self.config.freq_loss_weight}")
        print_rank_0(f"{'混合精度:':<15} {'启用' if self.use_amp else '禁用'}")
        print_rank_0(f"{'梯度累积步数:':<15} {self.gradient_accumulation_steps}")
        print_rank_0(f"{'每个周期的步数:':<15} {self.steps_per_epoch}")
        print_rank_0("="*50 + "\n")
        
        # =================================================================
        # === 步骤 2: 统一初始化所有计时器（避免重复时间获取）===
        # =================================================================
        if self.training_start_time is None:  # 只在首次训练时初始化
            self.training_start_time = time.time()
        self.last_periodic_save_time = self.training_start_time  # 重置周期性保存计时器
        print_rank_0(f"⏰ 训练计时器已初始化，周期性保存间隔: {checkpoint_interval_hours}小时")
        # =================================================================
        avg_train_loss = float('nan')
        avg_time_loss = float('nan')
        avg_freq_loss = float('nan')
        avg_phase_loss = float('nan')
        for epoch in range(num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch_start_time = time.time()
            epoch_losses = []
            epoch_time_losses = []
            epoch_freq_losses = []
            epoch_phase_losses = []
            print_rank_0(f"\nEpoch {epoch+1}/{num_epochs}")
            print_rank_0("-"*50)
            total_batches = len(train_loader)
            progress_step = max(1, total_batches // 20)
            try:
                for batch_idx, batch in enumerate(train_loader):
                    metrics = self.train_step(batch)
                    epoch_losses.append(metrics['loss'])
                    epoch_time_losses.append(metrics['time_loss'])
                    epoch_freq_losses.append(metrics['freq_loss'])
                    epoch_phase_losses.append(metrics['phase_loss'])
                    if is_main_process() and (batch_idx % progress_step == 0 or batch_idx == total_batches - 1):
                        progress = min(20, int(batch_idx * 20 / total_batches) + 1)
                        elapsed = time.time() - epoch_start_time
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print_rank_0(
                            f"\r训练进度: [{'=' * progress}{' ' * (20-progress)}] {batch_idx+1}/{total_batches} | "
                            f"总损失: {metrics['loss']:.8f} | 时域: {metrics['time_loss']:.8f} | 频域: {metrics['freq_loss']:.8f} | "
                            f"LR: {current_lr:.8f} | 用时: {elapsed:.1f}秒", end=""
                        )
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # =================================================================
                    # === 步骤 3: 嵌入基于时间的周期性检查点逻辑 ===
                    # =================================================================
                    # 1. 每隔 N 步，才检查一次时间，以降低开销
                    is_check_step = (self.global_step % check_time_every_n_steps == 0)

                    # 2. 创建一个信号张量，用于在所有进程间同步"是否保存"的决策
                    save_signal = torch.tensor(0, device=self.device)
                    
                    # 3. 只有主进程(rank 0)检查时间并做出决策
                    if self.is_ddp and self.rank == 0 and is_check_step:
                        current_time = time.time()
                        if (current_time - self.last_periodic_save_time) >= checkpoint_interval_seconds:
                            save_signal[0] = 1 # 决策：是时候保存了！

                    # (在非DDP模式下，也进行检查)
                    if not self.is_ddp and is_check_step:
                         current_time = time.time()
                         if (current_time - self.last_periodic_save_time) >= checkpoint_interval_seconds:
                            save_signal[0] = 1

                    # 4. 如果是DDP环境，将 rank 0 的决策广播给所有其他进程
                    if self.is_ddp:
                        dist.broadcast(save_signal, src=0)
                    
                    # 5. 如果信号为1 (需要保存)，则所有进程都执行相应操作
                    if save_signal.item() == 1:
                        # 只有主进程负责写入文件
                        if is_main_process():
                            periodic_ckpt_path = f"{self.run_checkpoint_dir}/periodic_checkpoint.pt"
                            print_rank_0(f"\n⏰ 时间达到 {checkpoint_interval_hours} 小时，正在保存周期性检查点到 {periodic_ckpt_path}...")
                            try:
                                self.save_checkpoint(periodic_ckpt_path)
                                # 关键：只有在成功保存后，才重置计时器
                                self.last_periodic_save_time = time.time()
                                elapsed_hours = (self.last_periodic_save_time - self.training_start_time) / 3600
                                print_rank_0(f"✅ 周期性检查点保存成功 (训练已进行 {elapsed_hours:.1f} 小时)")
                            except Exception as e:
                                print_rank_0(f"❌ 保存周期性检查点时发生错误: {e}")
                        
                        # 6. 设置同步栅栏，确保所有进程都等待 rank 0 保存完毕后再继续
                        if self.is_ddp:
                            dist.barrier()
                    # =================================================================
            except torch.cuda.OutOfMemoryError as e:
                print_rank_0(f"\n❌ CUDA显存不足错误: {str(e)}")
                print_rank_0("❌ 无法继续训练，程序将终止以防止后续错误")
                torch.cuda.empty_cache()
                gc.collect()
                raise e
            except Exception as e:
                print_rank_0(f"\n⚠️ 训练期间发生其他错误: {str(e)}")
                print_rank_0("🔄 尝试继续训练...")
                torch.cuda.empty_cache()
                gc.collect()
                self.optimizer.zero_grad()
                if self.use_amp and self.scaler is not None:
                    try:
                        # 重新初始化scaler以清除可能的错误状态
                        self.scaler = GradScaler(init_scale=2**16)
                        print_rank_0("scaler已重新初始化")
                    except Exception as scaler_error:
                        print_rank_0(f"警告: scaler重新初始化失败: {str(scaler_error)}")
                torch.cuda.synchronize()
                continue
            # 1. 统计本地批次数和损失和
            local_count = torch.tensor([len(epoch_losses)], dtype=torch.float32, device=self.device)
            local_loss = torch.tensor([sum(epoch_losses) if len(epoch_losses) > 0 else 0.0], dtype=torch.float32, device=self.device)
            local_time_loss = torch.tensor([sum(epoch_time_losses) if len(epoch_time_losses) > 0 else 0.0], dtype=torch.float32, device=self.device)
            local_freq_loss = torch.tensor([sum(epoch_freq_losses) if len(epoch_freq_losses) > 0 else 0.0], dtype=torch.float32, device=self.device)
            local_phase_loss = torch.tensor([sum(epoch_phase_losses) if len(epoch_phase_losses) > 0 else 0.0], dtype=torch.float32, device=self.device)
            local_stats = torch.cat([local_count, local_loss, local_time_loss, local_freq_loss, local_phase_loss], dim=0)
            if dist.is_initialized():
                dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
            total_count = local_stats[0].item()
            total_loss = local_stats[1].item()
            total_time_loss = local_stats[2].item()
            total_freq_loss = local_stats[3].item()
            total_phase_loss = local_stats[4].item()
            if total_count > 0:
                avg_train_loss = total_loss / total_count
                avg_time_loss = total_time_loss / total_count
                avg_freq_loss = total_freq_loss / total_count
                avg_phase_loss = total_phase_loss / total_count
            else:
                avg_train_loss = float('nan')
                avg_time_loss = float('nan')
                avg_freq_loss = float('nan')
                avg_phase_loss = float('nan')
            if is_main_process():
                print_rank_0("\n" + "-"*50)
                print_rank_0(f"{'Epoch 总结':^50}")
                print_rank_0(f"{'训练总损失:':<15} {avg_train_loss:.8f}")
                print_rank_0(f"{'训练时域损失:':<15} {avg_time_loss:.8f}")
                print_rank_0(f"{'训练频域损失:':<15} {avg_freq_loss:.8f}")
                print_rank_0(f"{'训练相位损失:':<15} {avg_phase_loss:.8f}")
                print_rank_0(f"{'当前学习率:':<15} {self.optimizer.param_groups[0]['lr']:.8f}")
                print_rank_0(f"{'本轮用时:':<15} {time.time() - epoch_start_time:.1f}秒")
                print_rank_0("-"*50)
            torch.cuda.empty_cache()
            gc.collect()
        total_training_time = time.time() - self.training_start_time
        hours, remainder = divmod(total_training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print_rank_0("\n" + "="*50)
        print_rank_0(f"{'训练完成':^50}")
        print_rank_0("-"*50)
        print_rank_0(f"{'总训练时间:':<15} {int(hours)}小时 {int(minutes)}分 {seconds:.1f}秒")
        final_loss_str = f"{avg_train_loss:.8f}" if not math.isnan(avg_train_loss) else "未计算 (无有效训练数据)"
        print_rank_0(f"{'最终训练损失:':<15} {final_loss_str}")
        print_rank_0(f"{'最佳训练损失:':<15} {self.best_loss:.8f}")
        print_rank_0(f"{'运行文件夹:':<15} {self.run_checkpoint_dir}")
        print_rank_0(f"{'最佳模型:':<15} {self.run_checkpoint_dir}/best_model.pt")
        print_rank_0(f"{'最后模型:':<15} {self.run_checkpoint_dir}/last_model.pt")
        print_rank_0(f"{'周期性检查点:':<15} {self.run_checkpoint_dir}/periodic_checkpoint.pt")
        print_rank_0("="*50)
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        return self.metrics
    
    def save_checkpoint(self, filename: str):
        """保存模型检查点"""
        
        # 只在主进程中执行保存操作，但所有进程都需等待
        if self.is_ddp:
            dist.barrier() # 确保所有进程同步

        if not is_main_process():
            return # 非主进程直接返回

        try:
            # 获取原始模型（去除DDP包装）
            raw_model = self.model.module if self.is_ddp else self.model

            # 创建检查点字典，直接使用 state_dict()，无需手动移至CPU
            checkpoint = {
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'global_step': self.global_step,
                'config': self.config, # 直接保存config对象
                'metrics': self.metrics,
                'best_loss': self.best_loss,
            }
            
            # 使用临时文件进行原子保存，防止写入中断导致文件损坏
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            temp_filename = filename + '.tmp'
            torch.save(checkpoint, temp_filename)
            
            # 原子性地重命名文件
            os.rename(temp_filename, filename)
            
            print_rank_0(f"✓ 模型检查点已成功保存到: {filename}")

        except Exception as e:
            print_rank_0(f"❌ 保存模型检查点失败: {str(e)}")
            # 尝试清理临时文件
            temp_filename = filename + '.tmp'
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass
            # 重新抛出异常，以便外部重试逻辑可以捕获
            raise e

        # === 修改/添加 ===
        # 步骤 2: 再次设置一个集合点，确保 rank 0 已经完成了保存操作，
        # 之后所有进程再一起安全地进入下一个训练周期。
        if dist.is_initialized():
            dist.barrier()
        # === 修改结束 ===
    
    def load_checkpoint(self, filename: str):
        """加载模型检查点"""
        if not os.path.exists(filename):
            print_rank_0(f"检查点文件 {filename} 不存在")
            return False
        
        checkpoint = torch.load(filename, map_location=self.device)
        
        # 根据当前是否使用DDP来加载模型状态
        if isinstance(self.model, DDP):
            # 当前使用DDP，直接加载到module
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 当前使用单GPU，直接加载
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.metrics = checkpoint['metrics']
        
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        
        if 'run_checkpoint_dir' in checkpoint:
            original_run_dir = checkpoint['run_checkpoint_dir']
            print_rank_0(f"原始运行文件夹: {original_run_dir}")
        
        # 显示分布式信息
        if 'is_ddp' in checkpoint and 'world_size' in checkpoint:
            original_is_ddp = checkpoint['is_ddp']
            original_world_size = checkpoint['world_size']
            print_rank_0(f"原始训练: {'DDP' if original_is_ddp else '单GPU'} (世界大小: {original_world_size})")
            current_world_size = dist.get_world_size() if self.is_ddp else 1
            print_rank_0(f"当前训练: {'DDP' if self.is_ddp else '单GPU'} (世界大小: {current_world_size})")
        
        print_rank_0(f"从 {filename} 加载检查点成功，全局步数: {self.global_step}，最佳损失: {self.best_loss:.8f}")
        return True

    def get_model(self) -> DualDomainTransformerMEM:
        """返回训练好的模型（去除DDP包装）"""
        return self.model.module if isinstance(self.model, DDP) else self.model


if __name__ == "__main__":
    # 尝试设置DDP环境
    try:
        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        print_rank_0(f"DDP初始化成功: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    except KeyError as e:
        print_rank_0(f"DDP环境变量缺失: {e}")
        print_rank_0("使用单GPU模式")
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 配置参数
    config = ModelConfig()
    
    # 输入参数 - 支持环境变量
    config.embed_dim = int(os.environ.get('EMBED_DIM', '256'))        # 嵌入维度
    config.num_heads = int(os.environ.get('NUM_HEADS', '8'))         # 注意力头数
    config.depth = int(os.environ.get('DEPTH', '2'))                 # Transformer层数

        
    # 正则化参数
    config.drop_rate = float(os.environ.get('DROP_RATE', '0.2'))     # dropout率
    config.attn_drop_rate = float(os.environ.get('ATTN_DROP_RATE', '0.2')) # 注意力dropout率
    config.drop_path_rate = float(os.environ.get('DROP_PATH_RATE', '0.2')) # 随机深度率

        
    # 初始化参数
    config.init_std = float(os.environ.get('INIT_STD', '0.02'))     # 初始化标准差
    config.use_abs_pos = os.environ.get('USE_ABS_POS', 'False').lower() == 'true'    # 是否使用绝对位置编码
    config.use_rel_pos = os.environ.get('USE_REL_POS', 'True').lower() == 'true'   # 是否使用相对位置编码
    config.use_time_embed = os.environ.get('USE_TIME_EMBED', 'True').lower() == 'true' # 是否使用时间嵌入
        
    # 掩码参数
    config.mask_ratio = float(os.environ.get('MASK_RATIO', '0.15'))    # 掩码比例
    config.mask_strategy = os.environ.get('MASK_STRATEGY', 'random')  # 掩码策略: random, block, structure
    config.mask_noise_ratio = float(os.environ.get('MASK_NOISE_RATIO', '0.005')) # 掩码噪声比例
        
    # MoE (Mixture of Experts) 相关参数
    config.use_moe = os.environ.get('USE_MOE', 'True').lower() == 'true'                 # 是否启用MoE替换FFN
    config.num_experts = int(os.environ.get('NUM_EXPERTS', '4'))                # 专家的数量
    config.top_k_experts = int(os.environ.get('TOP_K_EXPERTS', '2'))              # 每个token激活的专家数量
    config.moe_aux_loss_coeff = float(os.environ.get('MOE_AUX_LOSS_COEFF', '0.01'))      # 负载均衡辅助损失的权重系数
        
    # 训练参数
    config.lr = float(os.environ.get('LEARNING_RATE', '1e-4'))            # 学习率
    config.weight_decay = float(os.environ.get('WEIGHT_DECAY', '1e-4'))  # 权重衰减
    config.warmup_epochs = int(os.environ.get('WARMUP_EPOCHS', '5'))     # 预热周期
    config.use_amp = os.environ.get('USE_AMP', 'True').lower() == 'true'        # 是否使用混合精度训练
    config.clip_grad = float(os.environ.get('CLIP_GRAD', '0.1'))     # 梯度裁剪阈值
    config.freq_eval = os.environ.get('FREQ_EVAL', 'True').lower() == 'true'     # 是否进行频域评估
        
    # 数据标准化参数
    config.use_layer_norm = os.environ.get('USE_LAYER_NORM', 'True').lower() == 'true'  # 是否使用层归一化
    config.use_batch_norm = os.environ.get('USE_BATCH_NORM', 'True').lower() == 'true'  # 启用批归一化
    config.eps = float(os.environ.get('EPS', '1e-8'))           # 数值稳定性参数
        
    # 通道嵌入参数
    config.use_channel_embed = os.environ.get('USE_CHANNEL_EMBED', 'True').lower() == 'true'    # 是否使用通道嵌入
    config.channel_embed_dim = int(os.environ.get('CHANNEL_EMBED_DIM', '32'))      # 通道嵌入维度
    config.num_brain_regions = int(os.environ.get('NUM_BRAIN_REGIONS', '5'))       # 脑区数量（额叶、中央区、顶叶、枕叶、颞叶）
        
    # 添加频域相关参数
    config.freq_mask_ratio = float(os.environ.get('FREQ_MASK_RATIO', '0.3'))  # 频域掩码比例
    config.time_loss_weight = float(os.environ.get('TIME_LOSS_WEIGHT', '0.9'))  # 默认值从 0.7 提高到 0.9
    config.freq_loss_weight = float(os.environ.get('FREQ_LOSS_WEIGHT', '0.1'))  # 默认值从 0.3 降低到 0.1
        
    # 添加分层GAT相关参数
    config.channel_gat_initial_dim = int(os.environ.get('CHANNEL_GAT_INITIAL_DIM', '32'))  # 初始电极嵌入维度
    config.intra_region_gat_heads = int(os.environ.get('INTRA_REGION_GAT_HEADS', '4'))    # 脑区内GAT注意力头数
    config.intra_region_gat_dim_per_head = int(os.environ.get('INTRA_REGION_GAT_DIM_PER_HEAD', '32'))  # 脑区内GAT每个头的维度
    config.region_agg_attention_dim = int(os.environ.get('REGION_AGG_ATTENTION_DIM', '64'))  # 脑区聚合注意力维度
    config.inter_region_gat_heads = int(os.environ.get('INTER_REGION_GAT_HEADS', '4'))    # 脑区间GAT注意力头数
    config.inter_region_gat_dim_per_head = int(os.environ.get('INTER_REGION_GAT_DIM_PER_HEAD', '64'))  # 脑区间GAT每个头的维度
    
    # 添加训练参数
    config.batch_size = int(os.environ.get('BATCH_SIZE', '32'))         # 批次大小
    config.debug = os.environ.get('DEBUG', 'False').lower() == 'true'           # 调试模式
    
    # 添加一些断言检查，确保参数设置合理
    assert config.batch_size > 0, "批次大小必须大于0"
    assert config.embed_dim % config.num_heads == 0, "嵌入维度必须是注意力头数的整数倍"
    
    # 打印设备信息
    print_rank_0(f"使用设备: {device}")
    
    # 打印GPU信息
    if torch.cuda.is_available():
        if rank == 0:
            num_gpus = torch.cuda.device_count()
            print_rank_0(f"可用GPU数量: {num_gpus}")
            for i in range(num_gpus):
                print_rank_0(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print_rank_0(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print_rank_0("CUDA不可用，使用CPU训练")
    
    # 打印配置信息
    print_rank_0("\n=== 模型配置信息 ===")
    if rank == 0:
        for key, value in config.__dict__.items():
            print_rank_0(f"{key}: {value}")
    
    # 步骤1：创建原始模型实例
    model = DualDomainTransformerMEM(config).to(device)
    print_rank_0(f"模型已创建并移动到设备: {device}")
    
    # 步骤1.5：应用torch.compile优化
    # try:
    #     print_rank_0("正在应用torch.compile优化...")
    #     model = torch.compile(model)
    #     print_rank_0("torch.compile优化已成功应用")
    # except Exception as e:
    #     print_rank_0(f"torch.compile优化失败: {str(e)}")
    #     print_rank_0("继续使用未优化的模型")
    
    # 步骤2：DDP封装（如果在DDP环境中）
    if dist.is_initialized():
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        print_rank_0(f"模型已使用DDP封装在设备 {local_rank}")
    else:
        ddp_model = model
        print_rank_0("单GPU模式，无需DDP封装")
    
    # 步骤3：创建优化器（使用DDP封装后的模型参数）
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    print_rank_0(f"优化器已创建，学习率: {config.lr}")
    
    # 步骤4：创建训练器实例
    trainer = DualDomainTrainer(ddp_model, optimizer, config, device, rank)
    print_rank_0("训练器已创建")
    
    # 计算模型参数量
    def count_parameters(model):
        """计算模型参数量"""
        # 获取原始模型（去除DDP包装）
        raw_model = model.module if isinstance(model, DDP) else model
        
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # 按模块统计参数量
        module_params = {}
        for name, module in raw_model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    module_params[name] = params
        
        return {
            'total': total_params,
            'trainable': trainable_params, 
            'frozen': frozen_params,
            'modules': module_params
        }
    
    # 打印模型参数统计
    param_stats = count_parameters(ddp_model)
    print_rank_0("\n" + "="*60)
    print_rank_0(f"{'模型参数统计':^60}")
    print_rank_0("="*60)
    print_rank_0(f"{'总参数量:':<20} {param_stats['total']:,}")
    print_rank_0(f"{'可训练参数:':<20} {param_stats['trainable']:,}")
    print_rank_0(f"{'冻结参数:':<20} {param_stats['frozen']:,}")
    print_rank_0(f"{'参数大小:':<20} {param_stats['total'] * 4 / 1024 / 1024:.2f} MB (FP32)")
    print_rank_0(f"{'参数大小:':<20} {param_stats['total'] * 2 / 1024 / 1024:.2f} MB (FP16)")
    
    # 打印主要模块的参数量（仅在主进程中）
    print_rank_0("\n" + "-"*60)
    print_rank_0(f"{'主要模块参数分布':^60}")
    print_rank_0("-"*60)
    
    # 按参数量排序并显示前10个最大的模块（仅在主进程中）
    if is_main_process():
        sorted_modules = sorted(param_stats['modules'].items(), key=lambda x: x[1], reverse=True)
        for i, (name, params) in enumerate(sorted_modules[:10]):
            percentage = params / param_stats['total'] * 100
            print_rank_0(f"{name:<40} {params:>10,} ({percentage:>5.1f}%)")
        
        if len(sorted_modules) > 10:
            remaining_params = sum(params for _, params in sorted_modules[10:])
            remaining_percentage = remaining_params / param_stats['total'] * 100
            print_rank_0(f"{'其他模块':<40} {remaining_params:>10,} ({remaining_percentage:>5.1f}%)")
    
    print_rank_0("="*60)
    
    # 准备数据 - 支持环境变量
    # 如果要使用合并文件，确保路径指向包含合并文件的目录
    data_path = os.environ.get('DATA_PATH', "E:\BFM")
    print_rank_0(f"\n正在加载数据: {data_path}")

    # 1. 实例化数据集，供 DDP Sampler 使用
    # 这一步对于正确计算数据集总长度和分布式采样至关重要
    dataset = EEGBrainRegionDataset(data_path)

    # 2. 创建 DDP Sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    # 3. 直接创建DataLoader，重用已有的数据集实例
    from load_slurm_1 import custom_collate_fn
    
    # 根据服务器配置优化数据加载器设置
    # 服务器配置：每GPU配32个CPU核心
    num_workers = int(os.environ.get('NUM_WORKERS', '8'))  # 默认8个worker，可通过环境变量调整
    
    # 安全检查：确保不超过合理范围
    if num_workers > 16:
        print_rank_0(f"警告: num_workers={num_workers} 可能过高，建议设置为16以下")
    
    print_rank_0(f"数据加载配置: num_workers={num_workers}, 服务器CPU核心充足")
    
    train_loader = DataLoader(
        dataset,  # 重用已创建的数据集实例，避免重复初始化
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=num_workers,  # 利用多进程加载提升性能
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True if num_workers > 0 else False,  # 保持worker进程活跃，减少创建开销
        prefetch_factor=2 if num_workers > 0 else 2  # 预取批次数量
    )
    print_rank_0(f"数据加载器已创建: 样本数={len(dataset)}, 批次大小={config.batch_size}")
    
    # 设置训练参数 - 支持环境变量
    num_epochs = int(os.environ.get('NUM_EPOCHS', '10'))
    log_dir = os.environ.get('LOG_DIR', "./logs")
    
    # 打印分隔线和训练信息
    print_rank_0("\n" + "="*50)
    print_rank_0(f"开始训练 DualDomain Neural Transformer MEM 模型")
    print_rank_0(f"共{num_epochs}个周期，批次大小:{config.batch_size}")
    print_rank_0("="*50 + "\n")
    
    # 完整训练循环
    try:
        metrics = trainer.train(
            train_loader,
            sampler=sampler,
            num_epochs=num_epochs,
            log_dir=log_dir
        )
        
        # 打印训练结果
        print_rank_0("\n" + "="*50)
        print_rank_0("训练完成！")
        if metrics and 'train_loss' in metrics and len(metrics['train_loss']) > 0:
            print_rank_0(f"最终训练损失: {metrics['train_loss'][-1]:.8f}")
        print_rank_0("="*50 + "\n")
        
    except KeyboardInterrupt:
        print_rank_0("\n\n训练被用户中断！正在保存当前模型...")
        last_path = f"{trainer.run_checkpoint_dir}/last_model.pt"
        
        # 添加重试机制
        max_retries = 3
        for retry in range(max_retries):
            try:
                trainer.save_checkpoint(last_path)
                print_rank_0(f"✓ 最后模型已保存为 {last_path}")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print_rank_0(f"⚠️ 保存模型失败 (尝试 {retry + 1}/{max_retries}): {str(e)}")
                    print_rank_0("🔄 等待1秒后重试...")
                    time.sleep(1)
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print_rank_0(f"❌ 保存模型最终失败: {str(e)}")
        
        print_rank_0(f"最佳模型位置: {trainer.run_checkpoint_dir}/best_model.pt (损失: {trainer.best_loss:.8f})")
        print_rank_0(f"运行文件夹: {trainer.run_checkpoint_dir}")
    
    # 清理DDP环境
    cleanup_ddp()
