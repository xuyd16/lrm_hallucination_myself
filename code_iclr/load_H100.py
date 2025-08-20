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
import torch.distributed as dist
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
    支持懒加载和全局样本分片的EEG数据集类 (最终优化版 V3 - 修复合并文件逻辑)
    """
    def __init__(self, h5_dir: str, brain_regions: Optional[Dict[str, List[str]]] = None, window_size: int = 1600):
        # ... (init parameters are the same) ...
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
        
        all_h5_files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
        if not all_h5_files:
            raise ValueError(f"在目录 {h5_dir} 未找到任何H5文件！")

        global_sample_map, channels_per_data_key = self._build_global_maps(all_h5_files)
        
        total_samples = len(global_sample_map)
        print_rank_0(f"全局扫描完成: 共找到 {len(all_h5_files)} 个文件, {total_samples} 个总样本。")
        
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            self.sample_map = global_sample_map[rank::world_size]
            print_rank_0(f"DDP已启用，对 {total_samples} 个样本进行分片。Rank {rank} 分配到 {len(self.sample_map)} 个样本。")
        else:
            self.sample_map = global_sample_map

        self.channel_indices = self._create_channel_indices(channels_per_data_key)

        if dist.is_initialized():
            dist.barrier()
        
        print_rank_0(f"数据集初始化完成: 当前进程样本数: {len(self.sample_map)}")

    def _build_global_maps(self, h5_files: List[str]):
        global_sample_map = []
        channels_per_data_key = {} # 键现在是唯一的 data_key
        stride = self.window_size // 4

        for h5_path in h5_files:
            try:
                with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
                    possible_dset_names = ['eeg', 'eeg_data', 'data']
                    dset = None
                    for name in possible_dset_names:
                        if name in f:
                            dset = f[name]
                            break
                    
                    # 场景1：处理标准文件 (找到了 'eeg' 等)
                    if dset is not None:
                        # 对于标准文件，data_key 就是文件路径本身
                        data_key = h5_path
                        if 'chOrder' in dset.attrs:
                            ch_order_raw = dset.attrs['chOrder']
                            channel_names = [ch.decode('utf-8') if isinstance(ch, bytes) else str(ch) for ch in ch_order_raw]
                            channels_per_data_key[data_key] = channel_names
                        else:
                            print_rank_0(f"警告: 标准文件 {os.path.basename(h5_path)} 缺少 'chOrder' 属性，已跳过。")
                            continue

                        if dset.ndim == 2:
                            total_length = dset.shape[1]
                            n_samples = (total_length - self.window_size) // stride + 1 if total_length >= self.window_size else 1
                        elif dset.ndim == 3:
                            n_samples = dset.shape[0]
                        else:
                            continue
                        
                        for sample_idx in range(n_samples):
                            global_sample_map.append((data_key, sample_idx))

                    # 场景2：处理合并文件 (没找到标准名，但文件内有其他数据集)
                    else:
                        dataset_names = [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]
                        if not dataset_names:
                            print_rank_0(f"警告：在文件 {os.path.basename(h5_path)} 中未找到任何数据集，已跳过。")
                            continue
                        
                        print_rank_0(f"检测到合并文件 {os.path.basename(h5_path)}，包含 {len(dataset_names)} 个数据集")
                        for dataset_name in dataset_names:
                            try:
                                dset_merged = f[dataset_name]
                                # 【关键修正】为合并文件中的每个数据集创建唯一的 data_key
                                data_key = f"{h5_path}#{dataset_name}"
                                
                                if 'chOrder' in dset_merged.attrs:
                                    ch_order_raw = dset_merged.attrs['chOrder']
                                    channel_names = [ch.decode('utf-8') if isinstance(ch, bytes) else str(ch) for ch in ch_order_raw]
                                    channels_per_data_key[data_key] = channel_names
                                else:
                                    print_rank_0(f"  警告: 合并文件中的数据集 {dataset_name} 缺少 'chOrder' 属性，已跳过")
                                    continue
                                
                                if dset_merged.ndim == 2:
                                    total_length = dset_merged.shape[1]
                                    n_samples = (total_length - self.window_size) // stride + 1 if total_length >= self.window_size else 1
                                elif dset_merged.ndim == 3:
                                    n_samples = dset_merged.shape[0]
                                else:
                                    continue
                                
                                for sample_idx in range(n_samples):
                                    # 【关键修正】在 sample_map 中存入唯一的 data_key
                                    global_sample_map.append((data_key, sample_idx))
                            except Exception as e_inner:
                                print_rank_0(f"  警告: 处理合并文件中的数据集 {dataset_name} 失败: {e_inner}")

            except Exception as e_outer:
                print_rank_0(f"警告：扫描文件 {os.path.basename(h5_path)} 失败，已跳过。原因: {e_outer}")
                continue
        return global_sample_map, channels_per_data_key

    def _create_channel_indices(self, channels_per_data_key: Dict[str, List[str]]):
        key_channel_indices = {}
        for data_key, channel_names in channels_per_data_key.items():
            region_indices = {}
            for region in self.region_order:
                region_electrodes = self.brain_regions.get(region, [])
                indices = [i for i, ch in enumerate(channel_names) for elec in region_electrodes if elec.lower() == ch.lower()]
                region_indices[region] = indices
            key_channel_indices[data_key] = region_indices
        return key_channel_indices

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 从 sample_map 中获取唯一的 data_key 和样本索引
        data_key, sample_in_file_idx = self.sample_map[idx]
        
        # 【关键修正】从 data_key 中解析出文件路径和数据集名
        if '#' in data_key:
            target_h5_file, dataset_name = data_key.split('#', 1)
        else:
            target_h5_file, dataset_name = data_key, None
        
        # 使用唯一的 data_key 来查找通道索引
        current_channel_indices = self.channel_indices[data_key]
        
        try:
            with h5py.File(target_h5_file, 'r', libver='latest', swmr=True) as f:
                # 【关键修正】根据解析出的 dataset_name 来获取数据集
                if dataset_name is not None:
                    dset = f[dataset_name]
                else:
                    dset = f.get('eeg') or f.get('eeg_data') or f.get('data')
                    if dset is None: raise ValueError("在标准文件中未找到有效数据集")
                
                # 数据切片逻辑
                if dset.ndim == 2:
                    # 2D数据：[channels, time_points]
                    start_time = sample_in_file_idx * (self.window_size // 4)
                    end_time = start_time + self.window_size
                    if end_time > dset.shape[1]:
                        start_time = max(0, dset.shape[1] - self.window_size)
                        end_time = start_time + self.window_size
                    raw_slice = dset[:, start_time:end_time]
                elif dset.ndim == 3:
                    # 3D数据：[samples, channels, time_points]
                    raw_slice = dset[sample_in_file_idx, :, :]
                else:
                    raise ValueError(f"不支持的数据维度: {dset.ndim}")
                
                # 确保数据长度匹配window_size
                current_length = raw_slice.shape[1]
                if current_length < self.window_size:
                    # 填充到目标长度
                    pad_width = ((0, 0), (0, self.window_size - current_length))
                    window_slice = np.pad(raw_slice, pad_width, mode='constant', constant_values=0)
                elif current_length > self.window_size:
                    # 截取到目标长度
                    window_slice = raw_slice[:, :self.window_size]
                else:
                    window_slice = raw_slice
                
                grouped_data = np.zeros((self.num_regions, self.max_electrodes_per_region, self.window_size), dtype=np.float32)
                padding_mask = np.ones((self.num_regions, self.max_electrodes_per_region))

                for region_idx, region_name in enumerate(self.region_order):
                    electrode_indices = current_channel_indices.get(region_name, [])
                    if electrode_indices:
                        num_electrodes = len(electrode_indices)
                        region_data = window_slice[electrode_indices]
                        if num_electrodes > 0:
                            n_elec_to_copy = min(num_electrodes, self.max_electrodes_per_region)
                            grouped_data[region_idx, :n_elec_to_copy] = region_data[:n_elec_to_copy]
                            padding_mask[region_idx, :n_elec_to_copy] = 0
                
                return torch.from_numpy(grouped_data).float(), torch.from_numpy(padding_mask).bool()

        except Exception as e:
            file_display_name = os.path.basename(target_h5_file)
            if dataset_name is not None:
                file_display_name += f"[{dataset_name}]"
            print_rank_0(f"错误: 加载 {file_display_name} 样本 {sample_in_file_idx} 失败: {e}")
            return torch.zeros((self.num_regions, self.max_electrodes_per_region, self.window_size)), \
                   torch.ones((self.num_regions, self.max_electrodes_per_region)).bool()


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
    test_dir = "E:/BFM"
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
