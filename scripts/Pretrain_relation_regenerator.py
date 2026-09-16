import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from models.augmentation.relation_augmenter import RelationAugmenter
from models.regeneration.relation_regenerator import (
    RelationGenerator, create_mask, normal_initialization, dr4sr_cross_entropy_loss
)

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)

class AugmentedDataset(Dataset):
    """增强数据集（支持多行为嵌入）"""
    def __init__(self, augmented_data, item2idx, config):
        self.config = config
        self.item2idx = item2idx
        self.max_seq_len = config.training.max_seq_len

        actual_items = [asin for asin, idx in item2idx.items() if asin != '<PAD>']
        self.num_items = len(actual_items)

        self.PAD = 0
        self.SOS = self.num_items + 1
        self.EOS = self.num_items + 2

        # 预构建统一查找表：兼容 str / 原始键 / int 三种格式，把每个 item 的 3 次查表降为 1 次
        lookup = {}
        for key, idx in item2idx.items():
            lookup[key] = idx
            skey = str(key)
            if skey not in lookup:
                lookup[skey] = idx
            if isinstance(key, (int, float)) and not isinstance(key, bool):
                ikey = int(key)
                if ikey not in lookup:
                    lookup[ikey] = idx
        self._lookup = lookup
        self._sos_eos_set = {self.SOS, self.EOS}

        # 多行为配置
        mb_cfg = getattr(config, "multi_behavior", None)
        self.use_behavior_emb = bool(mb_cfg and getattr(mb_cfg, "enabled", False))
        self.num_behaviors = int(getattr(mb_cfg, "num_behaviors", 4)) if mb_cfg else 4

        logger.info(f"特殊token定义: PAD={self.PAD}, SOS={self.SOS}, EOS={self.EOS}")
        logger.info(f"实际物品数量: {self.num_items}")
        logger.info(f"多行为模式: {self.use_behavior_emb}")

        self.source, self.target = [], []
        self.source_seqlen, self.target_seqlen = [], []
        # 多行为：存储源序列的行为类型张量
        self.source_behaviors = [] if self.use_behavior_emb else None
        # 多行为：目标序列的行为类型张量（用于 behavior_head 的训练标签）
        # 增强数据没有显式标注 augmented_behaviors，这里用 original_behaviors 对齐到
        # target 长度做近似（截断/补零）。behavior_head 学到的是"给定 item 生成的位置
        # 应该是什么行为"的粗粒度模式，精确的行为标签生成留作 future work。
        self.target_behaviors = [] if self.use_behavior_emb else None

        self.length_changes = []
        self.original_lengths = []
        self.augmented_lengths = []

        if isinstance(augmented_data, dict) and 'pairs' in augmented_data:
            pairs_data = augmented_data['pairs']
            logger.info(f"检测到新格式增强数据，包含{len(pairs_data)}个数据对")
        elif isinstance(augmented_data, list):
            pairs_data = augmented_data
            logger.info(f"检测到旧格式增强数据，包含{len(pairs_data)}个数据对")
        else:
            raise ValueError(f"未知的增强数据格式: {type(augmented_data)}")

        logger.info(f"开始处理{len(pairs_data)}个增强数据对...")

        valid_pairs = 0
        invalid_pairs = 0
        empty_indices_pairs = 0

        for i, pair in enumerate(tqdm(pairs_data, desc="转换增强数据")):
            try:
                original_seq, augmented_seq = None, None
                original_behaviors = None

                if isinstance(pair, dict):
                    original_seq = pair.get('original', [])
                    augmented_seq = pair.get('augmented', [])
                    # 多行为：从 pair 中获取行为类型列表
                    original_behaviors = pair.get('original_behaviors', None)
                elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    original_seq, augmented_seq = pair[0], pair[1]
                elif isinstance(pair, (list, tuple)) and len(pair) == 1:
                    original_seq = augmented_seq = pair[0]
                else:
                    invalid_pairs += 1
                    continue

                if not original_seq or not augmented_seq:
                    invalid_pairs += 1
                    continue

                orig_len = len(original_seq)
                aug_len = len(augmented_seq)
                self.original_lengths.append(orig_len)
                self.augmented_lengths.append(aug_len)
                self.length_changes.append(aug_len - orig_len)

                src_indices = self._convert_to_indices(original_seq)
                tgt_indices = self._convert_to_indices(augmented_seq)

                if len(src_indices) < 1 or len(tgt_indices) < 1:
                    empty_indices_pairs += 1
                    self.original_lengths.pop()
                    self.augmented_lengths.pop()
                    self.length_changes.pop()
                    continue

                src_with_tokens = torch.tensor([self.SOS] + src_indices + [self.EOS])
                tgt_with_tokens = torch.tensor([self.SOS] + tgt_indices + [self.EOS])

                self.source.append(src_with_tokens)
                self.target.append(tgt_with_tokens)
                self.source_seqlen.append(len(src_with_tokens))
                self.target_seqlen.append(len(tgt_with_tokens))

                # 多行为：构建行为类型张量 [SOS用0填充, behaviors..., EOS用0填充]
                if self.use_behavior_emb and self.source_behaviors is not None:
                    if original_behaviors and len(original_behaviors) == len(original_seq):
                        # 裁剪到 src_indices 对应的有效长度
                        valid_behaviors = original_behaviors[:len(src_indices)]
                        beh_tensor = torch.tensor(
                            [0] + [int(b) % self.num_behaviors for b in valid_behaviors] + [0],
                            dtype=torch.long
                        )
                    else:
                        # 无行为信息时，默认全部为 purchase(2)
                        beh_tensor = torch.tensor(
                            [0] + [2] * len(src_indices) + [0], dtype=torch.long
                        )
                    self.source_behaviors.append(beh_tensor)

                    # 目标行为：用 source_behaviors 近似（截断/补零到 target 长度）
                    # target = [SOS] + augmented_items + [EOS]，行为对齐为 [0] + beh + [0]
                    tgt_inner_len = len(tgt_indices)
                    src_beh_inner = beh_tensor[1:-1].tolist()  # 去掉 SOS/EOS 的 0
                    if len(src_beh_inner) >= tgt_inner_len:
                        tgt_beh_inner = src_beh_inner[:tgt_inner_len]
                    else:
                        tgt_beh_inner = src_beh_inner + [0] * (tgt_inner_len - len(src_beh_inner))
                    tgt_beh_tensor = torch.tensor(
                        [0] + tgt_beh_inner + [0], dtype=torch.long
                    )
                    self.target_behaviors.append(tgt_beh_tensor)

                valid_pairs += 1

            except Exception as e:
                invalid_pairs += 1
                if i < 10:
                    logger.error(f"处理数据对{i}时出错: {e}")
                continue

        logger.info(f"数据处理完成: 有效={valid_pairs}, 无效={invalid_pairs}, 索引失败={empty_indices_pairs}")

        if self.length_changes:
            logger.info(f"序列长度变化统计:")
            logger.info(f"  平均原始长度: {np.mean(self.original_lengths):.2f}")
            logger.info(f"  平均增强长度: {np.mean(self.augmented_lengths):.2f}")
            logger.info(f"  平均长度变化: {np.mean(self.length_changes):.2f}")
            shorter = sum(1 for x in self.length_changes if x < 0)
            same    = sum(1 for x in self.length_changes if x == 0)
            longer  = sum(1 for x in self.length_changes if x > 0)
            total   = len(self.length_changes)
            logger.info(f"  变短/不变/变长: {shorter}({shorter/total*100:.1f}%) / {same}({same/total*100:.1f}%) / {longer}({longer/total*100:.1f}%)")

        if len(self.source) == 0:
            raise ValueError("没有有效的训练数据！请检查增强数据格式和item2idx映射。")

        self.source = pad_sequence(self.source, batch_first=True, padding_value=self.PAD)
        self.target = pad_sequence(self.target, batch_first=True, padding_value=self.PAD)

        if self.target.shape[1] < 20:
            padding_size = 20 - self.target.shape[1]
            self.target = torch.cat([
                self.target,
                torch.zeros(self.target.shape[0], padding_size, dtype=torch.long)
            ], dim=-1)

        # 对齐多行为张量的长度（与 source 对齐）
        if self.use_behavior_emb and self.source_behaviors:
            self.source_behaviors = pad_sequence(
                self.source_behaviors, batch_first=True, padding_value=0
            )
            # 裁剪或补齐到与 source 相同长度
            src_len = self.source.shape[1]
            beh_len = self.source_behaviors.shape[1]
            if beh_len < src_len:
                self.source_behaviors = torch.cat([
                    self.source_behaviors,
                    torch.zeros(self.source_behaviors.shape[0], src_len - beh_len, dtype=torch.long)
                ], dim=1)
            elif beh_len > src_len:
                self.source_behaviors = self.source_behaviors[:, :src_len]

            # target_behaviors 对齐到 target 长度
            self.target_behaviors = pad_sequence(
                self.target_behaviors, batch_first=True, padding_value=0
            )
            tgt_len = self.target.shape[1]
            tgt_beh_len = self.target_behaviors.shape[1]
            if tgt_beh_len < tgt_len:
                self.target_behaviors = torch.cat([
                    self.target_behaviors,
                    torch.zeros(self.target_behaviors.shape[0], tgt_len - tgt_beh_len, dtype=torch.long)
                ], dim=1)
            elif tgt_beh_len > tgt_len:
                self.target_behaviors = self.target_behaviors[:, :tgt_len]
        else:
            self.source_behaviors = None
            self.target_behaviors = None

        self.source_seqlen = torch.tensor(self.source_seqlen)
        self.target_seqlen = torch.tensor(self.target_seqlen)

        logger.info(f"最终数据集: {len(self.source)}个样本")
        logger.info(f"源序列形状: {self.source.shape}, 目标序列形状: {self.target.shape}")
        if self.source_behaviors is not None:
            logger.info(f"行为序列形状: source_behaviors={self.source_behaviors.shape}, target_behaviors={self.target_behaviors.shape}")

    def _convert_to_indices(self, sequence):
        """转换序列为索引 - 优化版（单次查表，无分支）"""
        lookup = self._lookup
        sos_eos = self._sos_eos_set
        out = []
        for item in sequence:
            idx = lookup.get(item)
            if idx is None:
                idx = lookup.get(str(item))
                if idx is None and isinstance(item, (int, float)) and not isinstance(item, bool):
                    idx = lookup.get(int(item))
            if idx is not None and idx > 0 and idx not in sos_eos:
                out.append(idx)
        return out

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        src_beh = (
            self.source_behaviors[index]
            if self.source_behaviors is not None
            else torch.zeros(self.source.shape[1], dtype=torch.long)
        )
        tgt_beh = (
            self.target_behaviors[index]
            if self.target_behaviors is not None
            else torch.zeros(self.target.shape[1], dtype=torch.long)
        )
        return (
            self.source[index],
            self.target[index],
            self.source_seqlen[index],
            self.target_seqlen[index],
            index,
            src_beh,   # 多行为：源序列行为类型张量
            tgt_beh,   # 多行为：目标序列行为类型张量（behavior_head 训练标签）
        )

def check_numerical_stability(tensor, name, batch_idx=None):
    """检查 tensor 是否含 NaN/Inf"""
    if tensor is None:
        return True, f"{name}: None"
    
    # 整数 tensor 先转 float 再检查 NaN/Inf
    if tensor.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
        check_tensor = tensor.float()
    else:
        check_tensor = tensor
    
    has_nan = torch.isnan(check_tensor).any().item()
    has_inf = torch.isinf(check_tensor).any().item()
    min_val = check_tensor.min().item() if check_tensor.numel() > 0 else 0
    max_val = check_tensor.max().item() if check_tensor.numel() > 0 else 0
    mean_val = check_tensor.mean().item() if check_tensor.numel() > 0 else 0
    
    prefix = f"Batch {batch_idx} - " if batch_idx is not None else ""
    status = f"{prefix}{name}: NaN={has_nan}, Inf={has_inf}, Range=[{min_val:.4f}, {max_val:.4f}], Mean={mean_val:.4f}"
    
    if has_nan or has_inf:
        logger.error(f"数值异常: {status}")
        return False, status
    else:
        return True, status

def train_epoch(model, dataloader, optimizer, scaler, device, config):
    """训练一个epoch（AMP混合精度 + 梯度累积）"""
    model.train()
    total_loss = 0
    num_batches = 0

    accum_steps = config.training.get('grad_accumulation_steps', 1)
    use_amp = config.training.get('use_amp', False)
    grad_clip = config.training.get('grad_clip', 0)

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(progress_bar):
        src, tgt, src_len, tgt_len, _, src_behaviors, tgt_behaviors = batch_data
        # pin_memory=True 已开，配合 non_blocking=True 实现异步 H2D 传输
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        src_len = src_len.to(device, non_blocking=True)
        tgt_len = tgt_len.to(device, non_blocking=True)
        src_behaviors = src_behaviors.to(device, non_blocking=True)
        tgt_behaviors = tgt_behaviors.to(device, non_blocking=True)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:, :-1])
        memory_key_padding_mask = src_padding_mask

        try:
            # 前向传播用 AMP 加速（fp16 节省显存），loss 计算强制 fp32
            # 避免大词表 logits 超出 fp16 范围（±65504）导致溢出
            # 兼容旧版 torch：新版 torch.amp.autocast 接收 device 参数，旧版回退 torch.cuda.amp.autocast
            if hasattr(torch.amp, 'autocast'):
                autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_ctx:
                logits, behavior_logits = model(
                    src=src,
                    tgt=tgt[:, :-1],
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    src_seqlen=src_len,
                    tgt_seqlen=tgt_len - 1,
                    relation_features=None,
                    src_behaviors=src_behaviors,
                    tgt_behaviors=tgt_behaviors[:, :-1],  # 对齐 tgt[:, :-1]
                )

            # 分块计算 CE：避免一次性物化 [N, V] 的 fp32 logits（大词表 V≈36万时
            # 全量 .float() 单张 4GB+，是 OOM 主因）。分块内 upcast，数值仍为 fp32。
            target = tgt[:, 1:]
            loss = dr4sr_cross_entropy_loss(logits, target, ignore_index=0, chunk_rows=512)

            # ── behavior_head 损失 ──────────────────────────────────────
            if behavior_logits is not None:
                behavior_logits_fp32 = behavior_logits.float()
                behavior_target = tgt_behaviors[:, 1:]  # 对齐：预测 t+1 位置的行为
                # behavior_target 的 padding 位置（=0）不参与 loss
                beh_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
                beh_loss = beh_loss_fn(
                    behavior_logits_fp32.reshape(-1, behavior_logits_fp32.size(-1)),
                    behavior_target.reshape(-1),
                )
                beh_loss_weight = config.training.get('behavior_loss_weight', 0.5)
                loss = loss + beh_loss_weight * beh_loss

            if hasattr(model.condition_encoder, 'condition4loss'):
                condition_prob = model.condition_encoder.condition4loss.float()
                condition_prob_clamped = torch.clamp(condition_prob, min=1e-12, max=1.0)
                reg_loss = -(condition_prob_clamped * torch.log(condition_prob_clamped)).sum(-1).mean()
                if not torch.isnan(reg_loss) and not torch.isinf(reg_loss):
                    loss = loss + config.training.get('diversity_weight', 0.1) * reg_loss

            # 梯度累积：对每个 micro-batch 的损失做缩放
            loss = loss / accum_steps

            # 单次 .item() 同步，复用于异常判断与显示，避免每 batch 多次 CPU-GPU 同步
            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val) or loss_val <= 0:
                # 跳过异常批次，仍需维持累积步数一致性
                if (batch_idx + 1) % accum_steps == 0:
                    optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            # 每累积 accum_steps 个 micro-batch 才更新一次参数
            if (batch_idx + 1) % accum_steps == 0:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss_val * accum_steps  # 还原为未缩放的损失值
            num_batches += 1

            progress_bar.set_postfix({
                'loss': f'{loss_val * accum_steps:.4f}',
                'avg':  f'{total_loss / num_batches:.4f}',
            })

        except Exception as e:
            if batch_idx < 3:
                logger.error(f"训练批次{batch_idx}出错: {e}")
            optimizer.zero_grad()
            continue

    if num_batches > 0:
        return total_loss / num_batches
    else:
        logger.error("没有成功的训练批次！")
        return float('inf')

@hydra.main(version_base=None, config_path="../configs", config_name="regeneration")
def main(cfg: DictConfig):
    """主训练函数"""
    # 设置随机种子
    seed = cfg.get('seed', 2023)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 速度优先：benchmark=True 让 cuDNN 自动挑选最快卷积/注意力算法
    # （Transformer 输入长度相对固定时收益明显）。如需严格可复现，把 deterministic 设为 True。
    deterministic = bool(cfg.get('deterministic', False))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Ampere+ GPU 上用 TF32 高精度矩阵乘，Transformer 训练显著提速
    torch.set_float32_matmul_precision('high')
    
    # 设置设备
    device = torch.device(cfg.resources.device)
    logger.info(f"使用设备: {device}")
    
    # 详细的数据加载过程
    logger.info("="*50)
    logger.info("开始加载数据...")
    
    # 1. 加载item2idx映射
    item2idx_path = Path(cfg.data.item2idx_path)
    logger.info(f"加载item2idx映射: {item2idx_path}")
    
    if not item2idx_path.exists():
        raise FileNotFoundError(f"item2idx文件不存在: {item2idx_path}")
    
    item2idx = torch.load(item2idx_path, weights_only=False)
    
    # 物品数不含 PAD
    actual_items = [asin for asin, idx in item2idx.items() if asin != '<PAD>']
    num_items = len(actual_items)
    
    logger.info(f"物品统计: 总计{len(item2idx)}个token, 实际物品{num_items}个")
    
    # 更新配置中的词汇表大小
    cfg.model.vocab_size = num_items + 3  # +3 for PAD, SOS, EOS
    
    logger.info(f"物品数量: {num_items}, 词汇表大小: {cfg.model.vocab_size}")
    
    # 2. 加载增强数据
    augmented_path = Path(cfg.data.augmented_path)
    logger.info(f"加载增强数据: {augmented_path}")
    
    if not augmented_path.exists():
        raise FileNotFoundError(f"增强数据文件不存在: {augmented_path}")
    
    # 验证文件大小
    file_size = augmented_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"增强数据文件大小: {file_size:.2f} MB")
    
    augmented_data = torch.load(augmented_path, weights_only=False)
    logger.info(f"增强数据类型: {type(augmented_data)}")
    from models.augmentation.relation_augmenter import (
        maybe_filter_augmented_payload,
        quality_control_from_config,
    )
    n_original = None
    sequences_path = cfg.data.get("sequences_path")
    if sequences_path and Path(sequences_path).exists():
        n_original = len(torch.load(sequences_path, map_location="cpu", weights_only=False))
    augmented_data, filt_stats = maybe_filter_augmented_payload(
        augmented_data,
        qc=quality_control_from_config(cfg),
        n_original=n_original,
    )
    if filt_stats:
        logger.info(f"增强质量过滤: {filt_stats['n_input']} -> {filt_stats['n_kept']}")
    
    # 详细分析数据结构
    if isinstance(augmented_data, dict):
        logger.info(f"增强数据字典键: {list(augmented_data.keys())}")
        if 'pairs' in augmented_data:
            logger.info(f"数据对数量: {len(augmented_data['pairs'])}")
            if 'metadata' in augmented_data:
                metadata = augmented_data['metadata']
                logger.info(f"元数据: {metadata}")
        else:
            logger.info(f"字典中各键的数据长度: {[(k, len(v) if hasattr(v, '__len__') else 'N/A') for k, v in augmented_data.items()]}")
    elif isinstance(augmented_data, list):
        logger.info(f"增强数据列表长度: {len(augmented_data)}")
        if len(augmented_data) > 0:
            logger.info(f"第一个元素类型: {type(augmented_data[0])}")
    
    # 3. 创建数据集和数据加载器
    logger.info("创建数据集...")
    dataset = AugmentedDataset(augmented_data, item2idx, cfg)
    
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    num_workers = cfg.resources.get('num_workers', 4)
    prefetch_factor = cfg.resources.get('prefetch_factor', 2)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        # 持久化 worker：100 个 epoch 不再每次重新 fork 进程并拷贝大数据集张量
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_init_fn
    )
    
    # 4. 创建模型
    logger.info("创建关系增强生成器...")
    
    # 在创建模型前记录关系图谱配置
    logger.info("关系图谱配置:")
    logger.info(f"  - 关系维度: {cfg.model.get('relation_dim', 0)}")
    logger.info(f"  - 关系模式: {cfg.model.get('relation_mode', 'disabled')}")
    logger.info(f"  - 融合策略: {cfg.model.get('fusion_strategy', 'none')}")
    
    model = RelationGenerator(
        config=cfg,
        num_items=num_items,
        sasrec_emb_path=cfg.paths.get('sasrec_emb_path')
    ).to(device)
    
    # 模型关系组件信息
    if hasattr(model, 'condition_encoder') and hasattr(model.condition_encoder, 'relation_dim'):
        logger.info(f"模型关系组件:")
        logger.info(f"  - 条件编码器关系维度: {model.condition_encoder.relation_dim}")
        logger.info(f"  - 多样性因子K: {model.K}")
        if hasattr(model.condition_encoder, 'relation_proj'):
            logger.info(f"  - 关系投影层: {model.condition_encoder.relation_proj}")
    
    # 通过配置文件参数，灵活控制再生器训练时的关系特征融合方式（“三者融合”或“仅PLM”），并在训练日志中明确记录当前采用的模式。
    feature_mode = getattr(cfg.model, "relation_feature_mode", "fused")
    if feature_mode == "plm_only":
        logger.info("【关系特征模式】仅使用PLM嵌入 (768->64维投影)")
    else:
        logger.info("【关系特征模式】使用PLM+共现+时序三者融合 (加权平均)")
    
    # 5. 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay
    )

    # AMP GradScaler（use_amp=False 时 scaler 不生效）
    use_amp = cfg.training.get('use_amp', False)
    # 兼容不同 torch 版本：新版用 torch.amp.GradScaler，旧版回退到 torch.cuda.amp.GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if hasattr(torch.amp, 'GradScaler') \
        else torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = cfg.training.get('grad_accumulation_steps', 1)
    logger.info(
        f"训练配置: batch_size={cfg.training.batch_size}, "
        f"grad_accumulation={accum_steps} (等效batch={cfg.training.batch_size * accum_steps}), "
        f"AMP={use_amp}"
    )

    # 6. 训练循环
    logger.info("开始训练...")
    logger.info("="*50)
    
    best_loss = float('inf')
    save_dir = Path(cfg.paths.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(cfg.training.epochs):
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        
        # 训练
        avg_loss = train_epoch(model, dataloader, optimizer, scaler, device, cfg)
        
        logger.info(f"Epoch {epoch+1} - 平均损失: {avg_loss:.4f}")
        
        # 如果损失异常，提前停止
        if math.isinf(avg_loss) or math.isnan(avg_loss):
            logger.error(f"训练损失异常，停止训练: {avg_loss}")
            break
        
        # 保存检查点
        if (epoch + 1) % cfg.training.save_interval == 0:
            checkpoint_path = save_dir / f"regenerator_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': cfg
            }, checkpoint_path)
            logger.info(f"保存检查点: {checkpoint_path}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = save_dir / "regenerator_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': cfg
            }, best_model_path)
            logger.info(f"保存最佳模型: {best_model_path} (loss: {avg_loss:.4f})")
    
    # 保存最终模型
    final_model_path = save_dir / "regenerator.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"训练完成，最终模型保存至: {final_model_path}")

if __name__ == "__main__":
    main()