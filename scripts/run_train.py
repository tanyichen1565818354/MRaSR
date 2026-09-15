# run_train.py - MRaSR baseline training script
import os
import torch
import logging
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from models.baseline.registry import get_model_class

import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.eval_utils import evaluate
import random
import time
import copy

def parse_args():
    """命令行参数解析（parse_known_args 让 Hydra 参数透传）"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device id')
    args, _ = parser.parse_known_args()
    return args

# 日志配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 确保日志不会重复输出
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(console_handler)

def calculate_sequence_stats(sequences):
    """计算序列统计信息"""
    if not sequences:
        return {"avg": 0, "min": 0, "max": 0, "total": 0}
    
    lengths = [len(seq) for seq in sequences]
    return {
        "avg": sum(lengths) / len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "total": len(sequences)
    }

def format_data_info(data, data_type="序列"):
    """格式化数据信息"""
    if isinstance(data, list):
        return f"列表, 包含{len(data)}个{data_type}"
    elif isinstance(data, dict):
        return f"字典, 包含{len(data)}个{data_type}"
    else:
        return f"未知格式({type(data).__name__})"

def show_sample_sequences(sequences, num_samples=3):
    """显示示例序列"""
    if not sequences:
        return "无可用示例"
    
    samples = []
    for i in range(min(num_samples, len(sequences))):
        seq = sequences[i]
        if len(seq) > 5:
            sample = f"{seq[:3]}...{seq[-2:]}"
        else:
            sample = str(seq)
        samples.append(sample)
    
    return ", ".join(samples)

def load_data(data_path, item_map_path=None):
    """加载预处理数据"""
    try:
        sequences = torch.load(data_path, weights_only=False)
        if item_map_path:
            item2idx = torch.load(item_map_path, weights_only=False)
        else:
            item2idx_path = os.path.join(os.path.dirname(data_path), "item2idx.pth")
            item2idx = torch.load(item2idx_path, weights_only=False)
        
        logger.info(f"Loaded {len(sequences)} sequences with {len(item2idx)} items")
        return sequences, item2idx
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def load_and_merge_data(cfg, split='train'):
    """根据精简配置加载并合并数据"""
    logger.info(f"开始加载{split}数据...")
    
    # 获取对应split的目录
    if split == 'train':
        data_dir = cfg.data.train_dir
    elif split == 'val':
        data_dir = cfg.data.val_dir
    elif split == 'test':
        data_dir = cfg.data.test_dir
    else:
        raise ValueError(f"不支持的数据split: {split}")
    
    # 加载基础数据
    base_sequences, item2idx = load_data(
        os.path.join(data_dir, "sequences.pth"), 
        cfg.data.item_map
    )
    
    # 如果是训练集，需要考虑数据合并
    if split == 'train':
        all_sequences = {}
        data_sources = []
        
        # 1. 处理原始数据（总是包含）
        all_sequences.update(base_sequences)
        data_sources.append('original')
        
        # 计算原始数据统计
        original_seqs = [data['sequence'] for data in base_sequences.values() if 'sequence' in data]
        stats = calculate_sequence_stats(original_seqs)
        logger.info(f"加载原始数据: {len(base_sequences)} 个序列")
        logger.info(f"原始序列长度统计: 平均={stats['avg']:.2f}, 最短={stats['min']}, 最长={stats['max']}")
        logger.info(f"示例原始序列: {show_sample_sequences(original_seqs[:3])}")
        
        # 2. 如果use_regen_augmented=True，加载再生和增强数据
        if cfg.data.get('use_regen_augmented', False):
            logger.info("检测到use_regen_augmented=True，开始加载再生和增强数据")
            
            # 加载再生数据
            regen_path = cfg.data.get('regen_path')
            logger.info(f"尝试加载再生数据: {regen_path}")
            if regen_path and os.path.exists(regen_path):
                try:
                    regen_data = torch.load(regen_path, weights_only=False)
                    logger.info(f"再生数据格式: {format_data_info(regen_data, '再生数据')}")
                    
                    # 转换再生数据
                    logger.info(f"开始转换再生数据")
                    regen_sequences, _ = convert_regenerated_data_format(regen_data, item2idx)
                    
                    if len(regen_sequences) > 0:
                        # 计算并显示序列统计
                        stats = calculate_sequence_stats(regen_sequences)
                        logger.info(f"再生序列长度统计: 平均={stats['avg']:.2f}, 最短={stats['min']}, 最长={stats['max']}")
                        logger.info(f"示例再生序列: {show_sample_sequences(regen_sequences[:3])}")
                        
                        # 合并数据
                        added_count = 0
                        skipped_count = 0
                        existing_sequences = set()
                        for existing_data in all_sequences.values():
                            existing_sequences.add(tuple(existing_data['sequence']))
                        
                        regen_start_id = max([int(uid) for uid in all_sequences.keys() if uid.isdigit()] + [0]) + 1
                        logger.info(f"开始合并{len(regen_sequences)}条再生序列...")
                        
                        for i, seq in enumerate(regen_sequences):
                            if (i + 1) % 10000 == 0:
                                logger.info(f"再生序列处理进度: {i+1}/{len(regen_sequences)}")
                            
                            user_id = f"regen_{regen_start_id + i}"
                            seq_tuple = tuple(seq)
                            
                            if seq_tuple not in existing_sequences:
                                all_sequences[user_id] = {'sequence': seq}
                                existing_sequences.add(seq_tuple)
                                added_count += 1
                            else:
                                skipped_count += 1
                        
                        data_sources.append('regenerated')
                        logger.info(f"成功添加{added_count}条再生序列到训练集")
                        if skipped_count > 0:
                            logger.info(f"跳过{skipped_count}条重复的再生序列")
                        logger.info(f"包含再生数据后训练集大小: {len(all_sequences)}条序列")
                    else:
                        logger.warning("再生数据转换后没有有效序列")
                        
                except Exception as e:
                    logger.warning(f"加载再生数据失败: {str(e)}")
                    import traceback
                    logger.warning(f"详细错误: {traceback.format_exc()}")
            else:
                logger.warning(f"再生数据文件不存在: {regen_path}")
            
            # 加载增强数据
            augmented_path = cfg.data.get('augmented_path')
            if augmented_path and os.path.exists(augmented_path):
                logger.info(f"尝试加载增强数据: {augmented_path}")
                try:
                    augmented_data = torch.load(augmented_path, weights_only=False)
                    logger.info(f"增强数据格式: {format_data_info(augmented_data, '增强数据')}")
                    from models.augmentation.relation_augmenter import (
                        maybe_filter_augmented_payload,
                        quality_control_from_config,
                    )
                    augmented_data, filt_stats = maybe_filter_augmented_payload(
                        augmented_data,
                        qc=quality_control_from_config(cfg),
                        n_original=len(base_sequences),
                    )
                    if filt_stats:
                        logger.info(
                            f"增强质量过滤: {filt_stats['n_input']} -> {filt_stats['n_kept']}"
                        )
                    
                    # 转换增强数据
                    logger.info(f"开始转换增强数据")
                    augmented_sequences, _ = convert_augmented_data_format(augmented_data, item2idx)
                    
                    if len(augmented_sequences) > 0:
                        # 计算并显示序列统计
                        stats = calculate_sequence_stats(augmented_sequences)
                        logger.info(f"增强序列长度统计: 平均={stats['avg']:.2f}, 最短={stats['min']}, 最长={stats['max']}")
                        logger.info(f"示例增强序列: {show_sample_sequences(augmented_sequences[:3])}")
                        
                        # 合并数据
                        added_count = 0
                        skipped_count = 0
                        existing_sequences = set()
                        for existing_data in all_sequences.values():
                            existing_sequences.add(tuple(existing_data['sequence']))
                        
                        aug_start_id = max([int(uid) for uid in all_sequences.keys() if uid.isdigit()] + [0]) + 1
                        logger.info(f"开始合并{len(augmented_sequences)}条增强序列...")
                        
                        for i, seq in enumerate(augmented_sequences):
                            if (i + 1) % 10000 == 0:
                                logger.info(f"增强序列处理进度: {i+1}/{len(augmented_sequences)}")
                            
                            user_id = f"aug_{aug_start_id + i}"
                            seq_tuple = tuple(seq)
                            
                            if seq_tuple not in existing_sequences:
                                all_sequences[user_id] = {'sequence': seq}
                                existing_sequences.add(seq_tuple)
                                added_count += 1
                            else:
                                skipped_count += 1
                        
                        data_sources.append('augmented')
                        logger.info(f"成功添加{added_count}条增强序列到训练集")
                        if skipped_count > 0:
                            logger.info(f"跳过{skipped_count}条重复的增强序列")
                        logger.info(f"包含增强数据后训练集大小: {len(all_sequences)}条序列")
                    else:
                        logger.warning("增强数据转换后没有有效序列")
                        
                except Exception as e:
                    logger.warning(f"加载增强数据失败: {str(e)}")
                    import traceback
                    logger.warning(f"详细错误: {traceback.format_exc()}")
            else:
                logger.warning(f"增强数据文件不存在: {augmented_path}")
        
        # 数据合并总结
        original_count = len(base_sequences)
        total_count_before = len(all_sequences)
        logger.info(f"总结数据合并情况:")
        logger.info(f"  - 原始序列: {original_count}条")

        # 训练集“按用户”子采样（仅对原始数据生效，保证可复现）
        train_ratio = float(cfg.data.get('train_ratio', 1))
        use_original_only = cfg.data.get('use_original_only', True)

        if split == 'train' and train_ratio < 0.999:
            if use_original_only:
                # 基于原始用户集合采样
                user_ids = sorted(base_sequences.keys())  # 用户ID集合（原始）
                rng = random.Random(cfg.get('seed', 2023))
                rng.shuffle(user_ids)

                keep_users = max(1, int(round(len(user_ids) * train_ratio)))
                kept_user_set = set(user_ids[:keep_users])

                # 只保留被选中用户的序列
                before_cnt = len(all_sequences)
                all_sequences = {uid: base_sequences[uid] for uid in kept_user_set if uid in base_sequences}
                logger.info(f"按用户采样训练集: ratio={train_ratio:.2f}, 原始用户数={len(user_ids)}, "
                            f"采样用户数={len(kept_user_set)}, 采样后序列={len(all_sequences)}")
            else:
                # 非纯原始数据时，aug/regen用户与原始用户无稳定映射，不建议按用户采样
                # 如需强行采样，这里退化为对所有键的确定性子采样（保留行为不变）
                total_before = len(all_sequences)
                keys = sorted(all_sequences.keys())
                rng = random.Random(cfg.get('seed', 2023))
                rng.shuffle(keys)
                keep = max(1, int(round(total_before * train_ratio)))
                kept = set(keys[:keep])
                all_sequences = {k: all_sequences[k] for k in kept}
                logger.info(f"按键采样训练集(非原始-only场景): ratio={train_ratio:.2f}, 原始={total_before}, 采样后={len(all_sequences)}")

        for source in data_sources:
            if source == 'original':
                continue
            elif source == 'regenerated':
                logger.info(f"  - 再生序列: 已合并到训练集")
            elif source == 'augmented':
                logger.info(f"  - 增强序列: 已合并到训练集")

        logger.info(f"  - 合并后总计(采样前): {total_count_before}条序列")
        if split == 'train':
            logger.info(f"  - 训练集采样后总计: {len(all_sequences)}条序列")
        logger.info(f"数据合并完成，数据源: {data_sources}")
        return all_sequences, item2idx
    
    else:
        # 验证集和测试集只使用原始数据
        return base_sequences, item2idx

def convert_regenerated_data_format(regen_data, item2idx):
    """Convert regenerated sequences into item-id lists used by baseline training."""
    sequences = []
    processed = 0
    skipped = 0

    idx2item = {idx: item for item, idx in item2idx.items()}
    # SOS index = num_items + 1 = len(item2idx) (item2idx includes PAD + N items)
    sos_idx = len(item2idx)

    if isinstance(regen_data, dict):
        if 'regenerated_pairs' in regen_data:
            pairs_data = regen_data['regenerated_pairs']
            logger.info(f"Detected regenerated_pairs format with {len(pairs_data)} pairs")

            for pair in pairs_data:
                processed += 1
                if isinstance(pair, dict):
                    generated_seq = None
                    if 'generated' in pair:
                        generated_seq = pair['generated']
                    elif 'augmented' in pair:
                        generated_seq = pair['augmented']
                    else:
                        for key in ['original', 'condition']:
                            if key in pair and isinstance(pair[key], list) and len(pair[key]) > 0:
                                generated_seq = pair[key]
                                break

                        if generated_seq is None:
                            skipped += 1
                            continue

                    if isinstance(generated_seq, list) and len(generated_seq) > 0:
                        if all(isinstance(x, (int, np.integer)) for x in generated_seq):
                            valid_seq = []
                            for idx in generated_seq:
                                # Skip special tokens (PAD=0, SOS=num_items+1, EOS=num_items+2)
                                if idx > 0 and idx < sos_idx and idx in idx2item:
                                    valid_seq.append(idx2item[idx])

                            if len(valid_seq) >= 2:
                                sequences.append(valid_seq)
                            else:
                                skipped += 1
                        else:
                            valid_seq = []
                            for item in generated_seq:
                                item_str = str(item)
                                if item_str in item2idx:
                                    valid_seq.append(item_str)
                                elif item in item2idx:
                                    valid_seq.append(item)

                            if len(valid_seq) >= 2:
                                sequences.append(valid_seq)
                            else:
                                skipped += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
        else:
            # 旧格式：直接的字典，每个键对应一个序列
            logger.info("检测到旧格式再生数据")
            for user_id, user_data in regen_data.items():
                processed += 1
                if isinstance(user_data, dict) and 'sequence' in user_data:
                    seq = user_data['sequence']
                    # 检查是否为索引格式
                    if all(isinstance(x, (int, np.integer)) for x in seq):
                        valid_seq = [idx2item[idx] for idx in seq if idx > 0 and idx < sos_idx and idx in idx2item]
                    else:
                        valid_seq = [str(item) for item in seq if str(item) in item2idx]
                    if len(valid_seq) >= 2:
                        sequences.append(valid_seq)
                    else:
                        skipped += 1
                elif isinstance(user_data, list):
                    # 检查是否为索引格式
                    if all(isinstance(x, (int, np.integer)) for x in user_data):
                        valid_seq = [idx2item[idx] for idx in user_data if idx > 0 and idx < sos_idx and idx in idx2item]
                    else:
                        valid_seq = [str(item) for item in user_data if str(item) in item2idx]
                    if len(valid_seq) >= 2:
                        sequences.append(valid_seq)
                    else:
                        skipped += 1
                else:
                    skipped += 1
                    
    elif isinstance(regen_data, list):
        # 如果是序列列表格式
        logger.info("检测到列表格式再生数据")
        for seq in regen_data:
            processed += 1
            if isinstance(seq, list) and len(seq) > 0:
                # 检查是否为索引格式
                if all(isinstance(x, (int, np.integer)) for x in seq):
                    valid_seq = [idx2item[idx] for idx in seq if idx > 0 and idx < sos_idx and idx in idx2item]
                else:
                    valid_seq = [str(item) for item in seq if str(item) in item2idx]
                
                if len(valid_seq) >= 2:  # 至少2个有效物品
                    sequences.append(valid_seq)
                else:
                    skipped += 1
            else:
                skipped += 1
    
    stats = {
        'processed': processed,
        'valid': len(sequences),
        'skipped': skipped
    }
    
    logger.info(f"转换再生数据: {len(sequences)} 个有效序列，跳过 {skipped} 个无效序列")
    return sequences, stats

def convert_augmented_data_format(augmented_data, item2idx):
    """转换增强数据格式"""
    sequences = []
    processed = 0
    skipped = 0
    
    # 检查数据格式并处理
    if isinstance(augmented_data, dict) and 'pairs' in augmented_data:
        # 新格式：包含pairs的字典
        pairs = augmented_data['pairs']
        for pair in pairs:
            processed += 1
            if isinstance(pair, dict) and 'augmented' in pair:
                seq = pair['augmented']
                if isinstance(seq, list) and len(seq) > 0:
                    # 验证序列中的物品是否在词汇表中
                    valid_seq = [item for item in seq if str(item) in item2idx]
                    if len(valid_seq) >= 2:  # 至少2个有效物品
                        sequences.append(valid_seq)
                    else:
                        skipped += 1
                else:
                    skipped += 1
            else:
                skipped += 1
    elif isinstance(augmented_data, list):
        # 如果是序列列表格式
        for seq in augmented_data:
            processed += 1
            if isinstance(seq, list) and len(seq) > 0:
                # 验证序列中的物品是否在词汇表中
                valid_seq = [item for item in seq if item in item2idx]
                if len(valid_seq) >= 2:  # 至少2个有效物品
                    sequences.append(valid_seq)
                else:
                    skipped += 1
            else:
                skipped += 1
    elif isinstance(augmented_data, dict):
        # 如果是字典格式
        for user_id, user_data in augmented_data.items():
            processed += 1
            if isinstance(user_data, dict) and 'sequence' in user_data:
                seq = user_data['sequence']
                valid_seq = [item for item in seq if item in item2idx]
                if len(valid_seq) >= 2:
                    sequences.append(valid_seq)
                else:
                    skipped += 1
            elif isinstance(user_data, list):
                valid_seq = [item for item in user_data if item in item2idx]
                if len(valid_seq) >= 2:
                    sequences.append(valid_seq)
                else:
                    skipped += 1
            else:
                skipped += 1
    
    stats = {
        'processed': processed,
        'valid': len(sequences),
        'skipped': skipped
    }
    
    logger.info(f"转换增强数据: {len(sequences)} 个有效序列，跳过 {skipped} 个无效序列")
    return sequences, stats

def worker_init_fn(worker_id):
    """确保DataLoader多进程的可重复性"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class LazyNextItemDataset(Dataset):
    """惰性 next-item 数据集：只存储对齐后的序列张量，按 flat index 即时生成样本。

    避免在内存中物化所有 (prefix, target) 样本——再生数据下 1.29M 条序列会展开为
    数千万个样本，预物化会 OOM。这里只存 [N, cap] 的序列（~1GB），__getitem__ 时
    按二分查找定位 (seq_idx, pos) 即时切片。

    注意：有效序列必须稠密存储，使 cum_counts 下标与 items 行下标一一对应；
    若按 dict 枚举下标存、跳过无效样本后留下空行，searchsorted 会指到 length=0
    的空槽，导致 input_len=-1、F.pad 出 [101] 等非法长度。
    """

    def __init__(self, sequences, item2idx, max_seq_len, is_test=False):
        self.max_seq_len = int(max_seq_len)
        self.is_test = is_test
        cap = self.max_seq_len + 1  # 训练需完整序列以生成所有前缀

        item_rows = []
        beh_rows = []
        uid_rows = []
        length_rows = []
        cum = []
        count = 0

        for seq_id, seq_data in sequences.items():
            if isinstance(seq_data, dict) and 'sequence' in seq_data:
                seq = seq_data['sequence']
                beh = seq_data.get('behaviors', None)
            elif isinstance(seq_data, list) and len(seq_data) > 0 and not isinstance(seq_data[0], (int, str)):
                continue
            else:
                seq = seq_data
                beh = None

            if len(seq) < 2:
                continue
            if any(item not in item2idx for item in seq):
                continue

            idx_seq = [item2idx[item] for item in seq]
            if beh is not None and len(beh) == len(seq):
                idx_beh = [int(b) + 1 for b in beh]
            else:
                idx_beh = [1] * len(seq)

            if len(idx_seq) > cap:
                idx_seq = idx_seq[-cap:]
                idx_beh = idx_beh[-cap:]

            l = len(idx_seq)
            row_item = torch.zeros(cap, dtype=torch.long)
            row_beh = torch.zeros(cap, dtype=torch.long)
            row_item[:l] = torch.tensor(idx_seq, dtype=torch.long)
            row_beh[:l] = torch.tensor(idx_beh, dtype=torch.long)
            item_rows.append(row_item)
            beh_rows.append(row_beh)
            length_rows.append(l)
            uid_rows.append(int(seq_id) if isinstance(seq_id, str) and seq_id.isdigit() else 0)

            count += 1 if is_test else (l - 1)
            cum.append(count)

        if not item_rows:
            self.items = torch.zeros(0, cap, dtype=torch.long)
            self.behs = torch.zeros(0, cap, dtype=torch.long)
            self.uids = torch.zeros(0, dtype=torch.long)
            self.lengths = torch.zeros(0, dtype=torch.long)
            self.cum_counts = torch.zeros(0, dtype=torch.long)
            self.total = 0
            return

        self.items = torch.stack(item_rows, dim=0)
        self.behs = torch.stack(beh_rows, dim=0)
        self.uids = torch.tensor(uid_rows, dtype=torch.long)
        self.lengths = torch.tensor(length_rows, dtype=torch.long)
        self.cum_counts = torch.tensor(cum, dtype=torch.long)
        self.total = count

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        seq_idx = torch.searchsorted(self.cum_counts, idx, right=True).item()
        prev = self.cum_counts[seq_idx - 1].item() if seq_idx > 0 else 0
        pos = idx - prev  # 0-indexed

        l = int(self.lengths[seq_idx].item())
        if self.is_test:
            # leave-one-out：输入为前 l-1，目标为最后一项
            input_len = min(l - 1, self.max_seq_len)
            start = max(0, (l - 1) - self.max_seq_len)
            in_items = self.items[seq_idx, start:start + input_len]
            in_behs = self.behs[seq_idx, start:start + input_len]
            target_item = self.items[seq_idx, l - 1]
            target_beh = self.behs[seq_idx, l - 1]
        else:
            input_len = pos + 1
            if input_len > self.max_seq_len:
                start = input_len - self.max_seq_len
                in_items = self.items[seq_idx, start:input_len]
                in_behs = self.behs[seq_idx, start:input_len]
                input_len = self.max_seq_len
            else:
                in_items = self.items[seq_idx, :input_len]
                in_behs = self.behs[seq_idx, :input_len]
            target_item = self.items[seq_idx, pos + 1]
            target_beh = self.behs[seq_idx, pos + 1]

        pad = self.max_seq_len - input_len
        if pad > 0:
            in_items = F.pad(in_items, (0, pad), value=0)
            in_behs = F.pad(in_behs, (0, pad), value=0)
        elif pad < 0:
            # 防御：绝不返回超过 max_seq_len 的张量
            in_items = in_items[-self.max_seq_len:]
            in_behs = in_behs[-self.max_seq_len:]
            input_len = self.max_seq_len

        return {
            'in_item_id': in_items,
            'in_behavior_id': in_behs,
            'item_id': target_item,
            'behavior_id': target_beh,
            'seqlen': input_len,
            'user_id': self.uids[seq_idx],
        }


def create_dataloader(sequences, item2idx, batch_size, max_seq_len=50, is_test=False, seed=2023,
                      num_workers=0, pin_memory=False, persistent_workers=False):
    """创建数据加载器（多行为版本，惰性样本生成）。

    使用 LazyNextItemDataset 只存储对齐后的序列张量，按 flat index 即时生成
    (prefix, target) 样本，避免再生数据下样本爆炸导致 OOM。
    """
    dataset = LazyNextItemDataset(sequences, item2idx, max_seq_len, is_test=is_test)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=generator,
    )

def collate_fn(batch):
    """批次整理函数（多行为版本，padding 行为 id 用 0）。"""
    max_len = 50  # 固定长度

    padded_inputs = []
    padded_behaviors = []
    targets = []
    target_behaviors = []
    seqlens = []
    user_ids = []

    for item in batch:
        input_seq = item['in_item_id']
        input_beh = item.get('in_behavior_id', [0] * len(input_seq))
        target_item = item['item_id']
        target_beh = item.get('behavior_id', 0)

        # 填充输入序列
        if len(input_seq) < max_len:
            pad = max_len - len(input_seq)
            padded_input = input_seq + [0] * pad
            padded_beh = input_beh + [0] * pad
        else:
            padded_input = input_seq[:max_len]
            padded_beh = input_beh[:max_len]

        padded_inputs.append(padded_input)
        padded_behaviors.append(padded_beh)
        targets.append(target_item)
        target_behaviors.append(target_beh)
        seqlens.append(item['seqlen'])
        user_ids.append(item['user_id'])

    return {
        'in_item_id': torch.LongTensor(padded_inputs),
        'in_behavior_id': torch.LongTensor(padded_behaviors),
        'item_id': torch.LongTensor(targets),
        'behavior_id': torch.LongTensor(target_behaviors),
        'seqlen': torch.LongTensor(seqlens),
        'user_id': torch.LongTensor(user_ids),
    }

# 早停策略类 - 直接在脚本中定义
class EarlyStopping:
    """DR4SR原版早停策略"""
    def __init__(self, model, monitor, dataset_name, cfg, save_dir='saved', 
                 patience=10, delta=0, mode='max', save_model=True):
        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_dir = save_dir
        self.save_model = save_model
        self.model_name = model.__class__.__name__
        self.cfg = cfg # 保存配置

        self._counter = 0
        self.best_value = np.inf if mode == 'min' else -np.inf
        
        if self.save_model:
            # 使用更通用的检查点名称
            self._best_ckpt_path = f"{self.model_name}/{dataset_name}/best_model.ckpt"
            self.best_ckpt = {
                'model': self.model_name,
                'epoch': 0,
                'parameters': copy.deepcopy(model.state_dict()),
                'metric': {monitor: np.inf if mode == 'min' else -np.inf},
                'config': self.cfg # 将配置添加到检查点
            }
            self.__check_save_dir()
        else:
            self.best_ckpt = None
            self._best_ckpt_path = None
    
    def __check_save_dir(self):
        if self.save_model and self.save_dir is not None:
            dir_path = os.path.dirname(os.path.join(self.save_dir, self._best_ckpt_path))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def __call__(self, model, epoch, metrics):
        if self.monitor not in metrics:
            raise ValueError(f"monitor {self.monitor} not in given metrics.")
        
        current_value = metrics[self.monitor]
        
        if self.mode == 'max':
            improved = current_value >= self.best_value + self.delta
        else:
            improved = current_value <= self.best_value - self.delta
        
        if improved:
            self._reset_counter(model, epoch, metrics)
            if self.save_model:
                self.save_checkpoint(epoch)
        else:
            self._counter += 1
        
        if self._counter >= self.patience:
            return True
        return False
    
    def _reset_counter(self, model, epoch, metrics):
        self._counter = 0
        self.best_value = metrics[self.monitor]
        if self.save_model:
            self.best_ckpt['parameters'] = copy.deepcopy(model.state_dict())
            self.best_ckpt['metric'] = metrics
            self.best_ckpt['epoch'] = epoch
            self.best_ckpt['config'] = self.cfg # 更新检查点中的配置

    def save_checkpoint(self, epoch):
        if self.save_model and self.save_dir is not None:
            self.save_path = os.path.join(self.save_dir, self._best_ckpt_path)
            torch.save(self.best_ckpt, self.save_path)

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Train a multi-behavior sequential recommender."""
    # 设置随机种子 - 确保完全可重复性
    seed = cfg.get("seed", 2023)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 确保完全确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"设置随机种子: {seed} (确保可重复性)")
    
    # 解析命令行参数
    args = parse_args()
    
    # 判断实验类型并设置日志文件名
    use_original_only = cfg.data.get('use_original_only', True)
    use_regen_augmented = cfg.data.get('use_regen_augmented', False)
    
    if use_original_only:
        log_filename = 'original_training_log.txt'
        experiment_type = "基线实验(仅原始数据)"
    elif use_regen_augmented:
        log_filename = 'regenk_augmented_training_log.txt'
        experiment_type = "增强实验(原始+再生+增强数据)"
    else:
        log_filename = 'training_log.txt'
        experiment_type = "未知实验类型"
    
    # 实验目录
    save_dir = os.path.abspath(cfg.training.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 清理已有的文件处理器，避免重复日志
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    
    # 日志配置
    log_file = os.path.join(save_dir, log_filename)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Start MRaSR baseline training")
    logger.info(f"配置:\n{OmegaConf.to_yaml(cfg)}")
    
    # 数据加载
    train_sequences, item2idx = load_and_merge_data(cfg, split='train')
    val_sequences, _ = load_and_merge_data(cfg, split='val')
    test_sequences, _ = load_and_merge_data(cfg, split='test')
    
    logger.info(f"数据加载完成：训练集 {len(train_sequences)}，验证集 {len(val_sequences)}，测试集 {len(test_sequences)}")
    
    # 数据加载
    num_workers = int(cfg.training.get('num_workers', 0))
    pin_memory = bool(cfg.training.get('pin_memory', False)) and torch.cuda.is_available()
    persistent_workers = bool(cfg.training.get('persistent_workers', False))
    train_loader = create_dataloader(train_sequences, item2idx, cfg.training.batch_size, cfg.data.max_seq_len,
                                     is_test=False, seed=seed, num_workers=num_workers,
                                     pin_memory=pin_memory, persistent_workers=persistent_workers)
    val_loader = create_dataloader(val_sequences, item2idx, cfg.training.batch_size, cfg.data.max_seq_len,
                                   is_test=True, seed=seed, num_workers=num_workers,
                                   pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = create_dataloader(test_sequences, item2idx, cfg.training.batch_size, cfg.data.max_seq_len,
                                    is_test=True, seed=seed, num_workers=num_workers,
                                    pin_memory=pin_memory, persistent_workers=persistent_workers)
    logger.info(f"DataLoader: num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")
    
    # 设备设置
    if torch.cuda.is_available():
        cuda_id = cfg.training.get('cuda_id', args.cuda_id)
        device = torch.device(f'cuda:{cuda_id}')
        torch.cuda.set_device(cuda_id)
        logger.info(f"使用GPU: {device}")
    else:
        device = torch.device('cpu')
        logger.warning("使用CPU训练")

    # 动态添加运行时参数到配置中
    OmegaConf.set_struct(cfg, False)
    cfg.num_items = len(item2idx)
    # 不要写 cfg.device = device
    OmegaConf.set_struct(cfg, True)
    
    # 动态获取模型类
    try:
        ModelClass = get_model_class(cfg.model.name)
        logger.info(f"使用模型: {cfg.model.name}")
    except (ValueError, ImportError) as e:
        logger.error(f"无法加载模型 '{cfg.model.name}': {e}")
        return

    # 模型初始化
    model = ModelClass(cfg).to(device)
    logger.info(f"模型初始化完成，词汇表大小: {len(item2idx)}")
    
    # 优化器 + 学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    # 多行为 + 稀疏目标行为下梯度方差大，固定 lr=0.001 训练到中后期会发散，
    # 因此在 val NDCG@20 停滞时自动把 lr 减半。patience 设小一点，让它在
    # early_stop patience(20) 用完之前先降 lr 把训练救回来。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=float(cfg.training.get('lr_scheduler_factor', 0.5)),
        patience=int(cfg.training.get('lr_scheduler_patience', 5)),
        min_lr=float(cfg.training.get('lr_scheduler_min_lr', 1e-5)),
    )
    # 梯度裁剪：base_config 已定义 clip_grad_norm=1.0，但原训练循环没接上，
    # 这是多行为 baseline 中后期发散的主因。
    clip_grad_norm = float(cfg.training.get('clip_grad_norm', 1.0))
    # 混合精度训练（AMP）：再生数据量大时，fp16 前向 + fp32 梯度可显著加速 GRU/LSTM/einsum
    use_amp = bool(cfg.training.get('use_amp', False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info(f"优化器: Adam(lr={cfg.training.learning_rate}, wd={cfg.training.weight_decay}); "
                f"ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5); "
                f"clip_grad_norm={clip_grad_norm}; amp={use_amp}")

    # 判断是否只使用原始数据，只有原始数据时才保存模型
    save_model = use_original_only  # 只有纯原始数据时才保存模型
    
    logger.info(f"实验类型: {experiment_type}")
    logger.info(f"日志文件: {log_filename}")
    logger.info(f"模型保存: {'是' if save_model else '否'}")

    # DR4SR原版早停策略
    early_stopping = EarlyStopping(
        model=model,
        monitor='ndcg@20',  # DR4SR原版监控ndcg@20
        dataset_name=cfg.data.name,  # 使用配置中的数据集名
        cfg=cfg, # 传入配置
        save_dir=save_dir,
        patience=cfg.training.early_stop_patience,
        mode='max',
        save_model=save_model  # 根据实验类型决定是否保存模型
    )
    logger.info(f'早停策略初始化完成，监控指标: ndcg@20, patience: {cfg.training.early_stop_patience}, 保存模型: {save_model}')

    # 训练循环 - DR4SR原版风格
    training_time = 0
    inference_time = 0
    
    try:
        for epoch in range(cfg.training.num_epochs):
            logged_metrics = {'epoch': epoch}
            
            # 训练阶段
            tik_train = time.time()
            model.train()
            total_loss = 0
            
            with tqdm(train_loader, desc=f"Training {epoch+1:>5}") as pbar:
                for batch in pbar:
                    batch = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items()}

                    # DR4SR原版负采样
                    batch['neg_item'] = model._neg_sampling(batch)
                    batch['neg_item'] = batch['neg_item'].to(device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss = model.training_step(batch)
                    scaler.scale(loss).backward()
                    # 梯度裁剪：多行为序列 + 稀疏 like 目标下梯度尺度波动大，
                    # 不裁剪会在中后期把 item_embedding 推飞（loss 从 0.28 涨到 0.72）。
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
            
            tok_train = time.time()
            training_time += tok_train - tik_train
            
            avg_loss = total_loss / len(train_loader)
            logged_metrics['train_loss'] = avg_loss
            
            # 验证阶段
            tik_valid = time.time()
            model.eval()
            val_metrics = evaluate(model, val_loader, k=20, device=device)  # DR4SR原版用k=20
            tok_valid = time.time()
            inference_time += tok_valid - tik_valid
            
            val_recall_20 = val_metrics['recall@20']
            val_ndcg_20 = val_metrics['ndcg@20']
            # 同时计算k=10的指标供参考
            val_metrics_10 = evaluate(model, val_loader, k=10, device=device)
            val_recall_10 = val_metrics_10['recall@10']
            val_ndcg_10 = val_metrics_10['ndcg@10']
            
            # 更新logged_metrics
            logged_metrics.update(val_metrics)  # 包含@20指标
            logged_metrics.update(val_metrics_10)  # 包含@10指标
            
            logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Recall@20: {val_recall_20:.4f} | NDCG@20: {val_ndcg_20:.4f}")
            logger.info(f"             | Recall@10: {val_recall_10:.4f} | NDCG@10: {val_ndcg_10:.4f}")
            logger.info(f'training_time: {training_time:.2f}s, inference_time: {inference_time:.2f}s, lr: {optimizer.param_groups[0]["lr"]:.2e}')

            # Per-epoch curve logging for paper Fig. 6 (training curves).
            # Enabled when training.curve_log_path is set in config.
            curve_path = cfg.training.get('curve_log_path', None)
            if curve_path:
                import csv as _csv
                _cp = Path(curve_path)
                if not _cp.is_absolute():
                    _cp = (root_dir.parent / _cp) if str(_cp).startswith("paper/") else (root_dir / _cp)
                _cp.parent.mkdir(parents=True, exist_ok=True)
                _new = not _cp.exists()
                with open(_cp, "a", newline="") as _f:
                    _w = _csv.writer(_f)
                    if _new:
                        _w.writerow(["epoch", "loss", "ndcg20"])
                    _w.writerow([epoch + 1, avg_loss, val_ndcg_20])

            # 学习率调度：监控 val NDCG@20，停滞则降 lr，防止中后期发散
            scheduler.step(val_ndcg_20)
            
            # DR4SR原版早停检查
            stop_training = early_stopping(model, epoch, logged_metrics)
            if stop_training:
                break
                
            # 定期测试
            if (epoch + 1) % 10 == 0 or epoch == cfg.training.num_epochs - 1:
                test_metrics = evaluate(model, test_loader, k=20, device=device)  # DR4SR原版用k=20
                test_metrics_10 = evaluate(model, test_loader, k=10, device=device)
                logger.info(f"Test: Recall@20: {test_metrics['recall@20']:.4f} | NDCG@20: {test_metrics['ndcg@20']:.4f}")
                logger.info(f"      Recall@10: {test_metrics_10['recall@10']:.4f} | NDCG@10: {test_metrics_10['ndcg@10']:.4f}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if save_model:
            early_stopping.save_checkpoint(epoch)
    
    # 最终评估
    # Optional per-user prediction dump for bucketed analysis (paper Fig. 11, Tables 9-11).
    # Enabled when eval.dump_path is set in config.
    dump_path = None
    if 'eval' in cfg and cfg.eval is not None:
        dump_path = cfg.eval.get('dump_path', None)
    if dump_path:
        _dp = Path(dump_path)
        if not _dp.is_absolute():
            _dp = (root_dir.parent / _dp) if str(_dp).startswith("paper/") else (root_dir / _dp)
        dump_path = str(_dp)

    if save_model and hasattr(early_stopping, 'save_path') and os.path.exists(early_stopping.save_path):
        # 有保存的最佳模型时，加载最佳模型进行评估
        logger.info(f"Loading best model from {early_stopping.save_path}")
        checkpoint = torch.load(early_stopping.save_path, weights_only=False)
        model.load_state_dict(checkpoint['parameters'])

        # 在测试集上评估最佳模型
        model.eval()
        if dump_path:
            from utils.eval_utils import evaluate_and_dump
            final_test_metrics_20 = evaluate_and_dump(model, test_loader, dump_path=dump_path, k=20, device=device)
            final_test_metrics_10 = evaluate(model, test_loader, k=10, device=device)
            logger.info(f"[dump] per-user predictions saved to {dump_path}")
        else:
            final_test_metrics_20 = evaluate(model, test_loader, k=20, device=device)
            final_test_metrics_10 = evaluate(model, test_loader, k=10, device=device)

        logger.info("="*80)
        logger.info(f"{experiment_type} - 最佳模型测试结果:")
        logger.info(f"Best epoch: {checkpoint['epoch']}")
        logger.info(f"Best validation NDCG@20: {checkpoint['metric']['ndcg@20']:.4f}")
        logger.info("-"*50)
        logger.info("测试集指标:")
        logger.info(f"Recall@20: {final_test_metrics_20['recall@20']:.4f}")
        logger.info(f"Recall@10: {final_test_metrics_10['recall@10']:.4f}")
        logger.info(f"NDCG@20: {final_test_metrics_20['ndcg@20']:.4f}")
        logger.info(f"NDCG@10: {final_test_metrics_10['ndcg@10']:.4f}")
        logger.info("="*80)
    else:
        # 没有落盘时，仍用 early_stopping 内存中的最佳权重做最终评估 / dump
        if early_stopping.best_ckpt is not None and early_stopping.best_ckpt.get('parameters') is not None:
            model.load_state_dict(early_stopping.best_ckpt['parameters'])
        model.eval()
        if dump_path:
            from utils.eval_utils import evaluate_and_dump
            final_test_metrics_20 = evaluate_and_dump(model, test_loader, dump_path=dump_path, k=20, device=device)
            final_test_metrics_10 = evaluate(model, test_loader, k=10, device=device)
            logger.info(f"[dump] per-user predictions saved to {dump_path}")
        else:
            final_test_metrics_20 = evaluate(model, test_loader, k=20, device=device)
            final_test_metrics_10 = evaluate(model, test_loader, k=10, device=device)
        logger.info("="*80)
        logger.info(f"{experiment_type} - 最终测试结果:")
        logger.info(f"最佳验证 NDCG@20: {early_stopping.best_value:.4f}")
        logger.info("-"*50)
        logger.info("测试集指标:")
        logger.info(f"Recall@20: {final_test_metrics_20['recall@20']:.4f}")
        logger.info(f"Recall@10: {final_test_metrics_10['recall@10']:.4f}")
        logger.info(f"NDCG@20: {final_test_metrics_20['ndcg@20']:.4f}")
        logger.info(f"NDCG@10: {final_test_metrics_10['ndcg@10']:.4f}")
        logger.info("="*80)

if __name__ == "__main__":
    main()