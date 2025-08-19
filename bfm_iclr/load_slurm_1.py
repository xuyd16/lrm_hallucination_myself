# -*- coding: utf-8 -*-
"""
高效懒加载版 EEG 数据加载器
适用于大规模H5文件数据集，支持PyTorch DDP分布式训练。
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import glob
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", message=".*Please report this")
warnings.filterwarnings("ignore", message=".*Performance may suffer")

def print_rank_0(message, end='\n'):
    """只在主分布式进程中打印消息，或在非分布式环境中打印。"""
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(message, end=end)
    except (ImportError, RuntimeError):
        print(message, end=end)

class EEGBrainRegionDataset(Dataset):
    """
    支持懒加载的EEG数据集类 (最终优化版)
    - 初始化时只扫描文件和元信息，不加载实际数据
    - 仅在__getitem__时才打开文件、读取单个样本
    """
    def __init__(self, h5_dir: str, brain_regions: Optional[Dict[str, List[str]]] = None, window_size: int = 1600):
        self.h5_dir = h5_dir
        self.window_size = window_size
        self.region_order = ['frontal', 'central', 'parietal', 'temporal', 'occipital']
        self.num_regions = 5
        self.max_electrodes_per_region = 24
        self.default_brain_regions = {
            'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'F1', 'F2', 'F5', 'F6', 'AF3', 'AF4', 'AF7', 'AF8', 'AFz', 'FT7', 'FT8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
            'central': ['C3', 'C4', 'Cz', 'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz'],
            'parietal': ['P3', 'P4', 'P7', 'P8', 'Pz', 'P1', 'P2', 'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'TP7', 'TP8', 'TP9'],
            'temporal': ['T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'FT7', 'FT8', 'TP7', 'TP8', 'TP9', 'TP10'],
            'occipital': ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'CB1', 'CB2']
        }
        self.brain_regions = brain_regions or self.default_brain_regions

        self.h5_files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
        if not self.h5_files:
            raise ValueError(f"在目录 {h5_dir} 未找到任何H5文件！")

        # 职责拆分：_build_sample_map_and_channels 负责扫描样本和通道名
        self.sample_map, channel_names_per_file = self._build_sample_map_and_channels()
        
        # 职责拆分：_create_channel_indices 负责计算索引
        # Bug修复：使用 self.channel_indices 而不是 self.file_channel_indices
        self.channel_indices = self._create_channel_indices(channel_names_per_file)
        
        # 统计实际文件数
        unique_files = set()
        for file_info in self.h5_files:
            if isinstance(file_info, tuple):
                unique_files.add(file_info[0])  # 文件路径
            else:
                unique_files.add(file_info)
        
        print_rank_0(f"数据集初始化完成: 共{len(unique_files)}个文件, {len(self.h5_files)}个数据源, 总样本数: {len(self.sample_map)}")

    def _build_sample_map_and_channels(self):
        sample_map = []
        channel_names_per_file = []
        valid_files = []  # 记录有效的文件
        stride = self.window_size // 4

        valid_file_idx = 0  # 使用单独的有效文件索引
        for original_file_idx, h5_path in enumerate(self.h5_files):
            try:
                with h5py.File(h5_path, 'r') as f:
                    # 查找数据和通道名 - 支持合并文件格式
                    dset = f.get('eeg') or f.get('eeg_data') or f.get('data')
                    
                    # 如果没有找到标准数据集名称，检查是否为合并文件
                    if dset is None:
                        # 获取所有数据集名称
                        dataset_names = [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]
                        if len(dataset_names) > 0:
                            # 对于合并文件，处理每个独立的数据集
                            print_rank_0(f"检测到合并文件 {os.path.basename(h5_path)}，包含 {len(dataset_names)} 个数据集")
                            
                            # 处理合并文件中的每个数据集
                            for dataset_name in dataset_names:
                                try:
                                    dset = f[dataset_name]
                                    
                                    # 获取通道名称（从属性中）
                                    if 'chOrder' in dset.attrs:
                                        ch_order_raw = dset.attrs['chOrder']
                                        channel_names = [ch.decode('utf-8') for ch in ch_order_raw] if hasattr(ch_order_raw, 'decode') else list(ch_order_raw)
                                    else:
                                        # 如果没有通道属性，使用默认命名
                                        n_ch = dset.shape[0] if dset.ndim > 1 else 1
                                        channel_names = [f'Ch{i+1}' for i in range(n_ch)]
                                    
                                    # 将每个数据集作为独立文件处理
                                    valid_files.append((h5_path, dataset_name))  # 存储文件路径和数据集名称
                                    channel_names_per_file.append(channel_names)
                                    
                                    # 计算样本数
                                    if dset.ndim == 2:
                                        total_length = dset.shape[1]
                                        if total_length < self.window_size:
                                            n_samples = 1
                                        else:
                                            n_samples = (total_length - self.window_size) // stride + 1
                                    elif dset.ndim == 3:
                                        n_samples = dset.shape[0]
                                    else:
                                        continue  # 跳过不支持的维度
                                    
                                    # 添加样本映射
                                    for sample_idx in range(n_samples):
                                        sample_map.append((valid_file_idx, sample_idx))
                                    
                                    valid_file_idx += 1
                                    #print_rank_0(f"  ✓ 处理数据集: {dataset_name}, 样本数: {n_samples}")
                                    
                                except Exception as e:
                                    print_rank_0(f"  ✗ 处理数据集 {dataset_name} 失败: {e}")
                                    continue
                            
                            # 合并文件处理完成，跳到下一个文件
                            continue
                        else:
                            raise ValueError("未找到有效的数据集")

                    if 'eeg' in f and 'chOrder' in f['eeg'].attrs:
                        ch_order_raw = f['eeg'].attrs['chOrder']
                        channel_names = [ch.decode('utf-8') for ch in ch_order_raw] if hasattr(ch_order_raw, 'decode') else list(ch_order_raw)
                    else:
                        channel_names_raw = next((f[key][:] for key in ['channel_names', 'channels', 'ch_names'] if key in f), None)
                        if channel_names_raw is not None:
                            channel_names = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in channel_names_raw]
                        else:
                            n_ch = dset.shape[0] if dset.ndim > 1 else 1
                            channel_names = [f'Ch{i+1}' for i in range(n_ch)]
                    
                    # 只有成功处理的文件才添加到列表中
                    valid_files.append((h5_path, None))  # 标准文件：(文件路径, None)
                    channel_names_per_file.append(channel_names)
                    
                    # 优化建议：修正 n_samples 的计算逻辑
                    if dset.ndim == 2:
                        total_length = dset.shape[1]
                        if total_length < self.window_size:
                            n_samples = 1
                        else:
                            n_samples = (total_length - self.window_size) // stride + 1
                    elif dset.ndim == 3:
                        n_samples = dset.shape[0]
                    else:
                        raise RuntimeError(f"文件 {h5_path} 的数据维度 {dset.ndim} 不支持")

                    # 使用有效文件索引而不是原始文件索引
                    for sample_idx in range(n_samples):
                        sample_map.append((valid_file_idx, sample_idx))
                    
                    valid_file_idx += 1  # 只有成功处理的文件才增加索引
                    
            except Exception as e:
                print_rank_0(f"警告：扫描文件 {os.path.basename(h5_path)} 失败，已跳过。原因: {e}")
                continue  # 跳过失败的文件，不增加valid_file_idx

        # 计算跳过的文件数量
        original_file_count = len(self.h5_files)
        
        # 统计有效的数据源（可能包含合并文件中的多个数据集）
        actual_data_sources = len(valid_files)
        
        # 统计实际处理的原始文件数量（去重）
        unique_files = set()
        for file_info in valid_files:
            if isinstance(file_info, tuple):
                unique_files.add(file_info[0])  # 文件路径
            else:
                unique_files.add(file_info)
        processed_file_count = len(unique_files)
        
        skipped_count = original_file_count - processed_file_count
        
        # 更新h5_files为只包含有效文件信息
        self.h5_files = valid_files
        print_rank_0(f"有效数据源: {actual_data_sources}, 来自 {processed_file_count} 个文件, 跳过文件数量: {skipped_count}")
        
        return sample_map, channel_names_per_file

    def _create_channel_indices(self, channel_names_per_file: List[List[str]]):
        file_channel_indices = {}
        for file_idx, channel_names in enumerate(channel_names_per_file):
            region_indices = {}
            for region in self.region_order:
                region_electrodes = self.brain_regions.get(region, [])
                indices = [i for i, ch in enumerate(channel_names) for elec in region_electrodes if elec.lower() == ch.lower()]
                region_indices[region] = indices
            file_channel_indices[file_idx] = region_indices
        return file_channel_indices

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 防御性检查：确保索引有效
        if idx >= len(self.sample_map):
            print_rank_0(f"警告: 索引 {idx} 超出样本映射范围 {len(self.sample_map)}")
            return torch.zeros((self.num_regions, self.max_electrodes_per_region, self.window_size)), \
                   torch.ones((self.num_regions, self.max_electrodes_per_region)).bool()
        
        file_idx, sample_in_file_idx = self.sample_map[idx]
        
        # 防御性检查：确保文件索引有效
        if file_idx >= len(self.h5_files):
            print_rank_0(f"警告: 文件索引 {file_idx} 超出文件列表范围 {len(self.h5_files)}")
            return torch.zeros((self.num_regions, self.max_electrodes_per_region, self.window_size)), \
                   torch.ones((self.num_regions, self.max_electrodes_per_region)).bool()
        
        # 防御性检查：确保通道索引存在
        if file_idx not in self.channel_indices:
            print_rank_0(f"警告: 文件索引 {file_idx} 不在通道索引字典中")
            return torch.zeros((self.num_regions, self.max_electrodes_per_region, self.window_size)), \
                   torch.ones((self.num_regions, self.max_electrodes_per_region)).bool()
        
        # 获取文件信息：(文件路径, 数据集名称或None)
        file_info = self.h5_files[file_idx]
        if isinstance(file_info, tuple):
            target_h5_file, dataset_name = file_info
        else:
            # 向后兼容：如果是字符串，则是标准格式
            target_h5_file, dataset_name = file_info, None

        try:
            with h5py.File(target_h5_file, 'r', libver='latest', swmr=True) as f:
                if dataset_name is not None:
                    # 合并文件：直接访问指定的数据集
                    dset = f[dataset_name]
                else:
                    # 标准文件：查找标准数据集名称
                    dset = f.get('eeg') or f.get('eeg_data') or f.get('data')
                    if dset is None: raise ValueError("未找到有效数据集")

                if dset.ndim == 2:
                    start_time = sample_in_file_idx * (self.window_size // 4)
                    end_time = start_time + self.window_size
                    if end_time > dset.shape[1]:
                        start_time = max(0, dset.shape[1] - self.window_size)
                    raw_slice = dset[:, start_time:end_time]
                else:
                    raw_slice = dset[sample_in_file_idx, :, :]
        except Exception as e:
            file_display_name = f"{os.path.basename(target_h5_file)}"
            if dataset_name is not None:
                file_display_name += f"[{dataset_name}]"
            print_rank_0(f"错误: 加载文件 {file_display_name} 样本 {sample_in_file_idx} 失败: {e}")
            return torch.zeros((self.num_regions, self.max_electrodes_per_region, self.window_size)), \
                   torch.ones((self.num_regions, self.max_electrodes_per_region)).bool()

        current_length = raw_slice.shape[1]
        if current_length < self.window_size:
            pad_width = ((0, 0), (0, self.window_size - current_length))
            window_slice = np.pad(raw_slice, pad_width, mode='constant', constant_values=0)
        elif current_length > self.window_size:
            window_slice = raw_slice[:, :self.window_size]
        else:
            window_slice = raw_slice

        assert window_slice.shape[1] == self.window_size, \
            f"错误：填充后窗口长度 ({window_slice.shape[1]}) 与目标 ({self.window_size}) 不匹配"

        grouped_data = np.zeros((self.num_regions, self.max_electrodes_per_region, self.window_size), dtype=np.float32)
        padding_mask = np.ones((self.num_regions, self.max_electrodes_per_region))

        for region_idx, region_name in enumerate(self.region_order):
            # Bug修复：使用 self.channel_indices
            electrode_indices = self.channel_indices[file_idx].get(region_name, [])
            if electrode_indices:
                num_electrodes = len(electrode_indices)
                region_data = window_slice[electrode_indices]
                if num_electrodes > 0:
                    n_elec_to_copy = min(num_electrodes, self.max_electrodes_per_region)
                    grouped_data[region_idx, :n_elec_to_copy] = region_data[:n_elec_to_copy]
                    padding_mask[region_idx, :n_elec_to_copy] = 0

        grouped_data_tensor = torch.from_numpy(grouped_data).float()
        padding_mask_tensor = torch.from_numpy(padding_mask).bool()
        return grouped_data_tensor, padding_mask_tensor

def custom_collate_fn(batch):
    if len(batch) == 0:
        return None
    data_list, mask_list = zip(*batch)
    data_batch = torch.stack(data_list, dim=0)
    mask_batch = torch.stack(mask_list, dim=0)
    return data_batch, mask_batch

def create_data_loader(h5_dir: str, batch_size: int = 1, brain_regions: Optional[Dict[str, List[str]]] = None, **kwargs) -> DataLoader:
    dataset = EEGBrainRegionDataset(h5_dir, brain_regions=brain_regions)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, **kwargs)
    print(f"数据加载器已创建: 样本数={len(dataset)}, 批次大小={batch_size}")
    return dataloader

if __name__ == "__main__":
    test_dir = "E:/BFM/processed_data"
    if os.path.exists(test_dir):
        dataset = EEGBrainRegionDataset(test_dir)
        print(f"总样本数: {len(dataset)}")
        data, mask = dataset[0]
        print(f"单样本数据形状: {data.shape}, 掩码形状: {mask.shape}")
        dataloader = create_data_loader(test_dir, batch_size=2)
        for i, batch in enumerate(dataloader):
            if batch is not None:
                data_batch, mask_batch = batch
                print(f"批次 {i}: 数据形状 {data_batch.shape}, 掩码形状 {mask_batch.shape}")
            if i >= 2:
                break
    else:
        print(f"测试路径 {test_dir} 不存在，请修改为实际数据路径")
