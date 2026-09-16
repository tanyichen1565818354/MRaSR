# scripts/generate_regenerated_data.py
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from models.regeneration.relation_regenerator import RelationGenerator
import time
import logging


logger = logging.getLogger(__name__)

def generate_square_subsequent_mask(sz, device='cuda'):
    """生成后续掩码（bool型，避免dtype警告）"""
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


# ──────────────────────────────────────────────
#  单序列函数（仅用于启动时的一次性测试）
# ──────────────────────────────────────────────
def _single_inference_mask(logits, src, ys):
    """单序列推理掩码（前2步）"""
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask = mask.scatter(-1, src, True)
    mask = mask.scatter(-1, ys, False)
    return logits.masked_fill(~mask, float('-inf'))

def _single_inference_mask_generative(logits, ys):
    """单序列生成掩码（2步后）"""
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask = mask.scatter(-1, ys, False)
    return logits.masked_fill(~mask, float('-inf'))


def _eos_logit_behavior0(model, decoder_out, eos_id):
    """EOS 用训练时的行为占位 0 走同一套 proj，再点积 e_EOS。

    直接用 decoder_out · e_EOS 会和双头物品 logit 不在同一尺度，
    第一步就可能把 EOS 选成 argmax，得到空序列。
    """
    h = decoder_out[:, -1]
    if getattr(model, "use_behavior_head", False) and getattr(model, "behavior_condition_proj", None) is not None:
        beh0 = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        beh_emb = model.behavior_embedding(beh0)
        h = model.behavior_condition_proj(torch.cat([h, beh_emb], dim=-1))
    return h @ model.item_embedding_decoder.weight[eos_id]


def _apply_eos_with_length_stop(logits, model, decoder_out, eos_id, ys, src, pad=0, step=0):
    """物品仍用双头；EOS 用行为 0 的 proj，并在接近/达到原序列长度时停止。

    全词表贪婪下 EOS 很难赢过上万物品，因此按源长度加分，达到源长度则强制 EOS。
    """
    logits = logits.clone()
    src_len = (src != pad).sum(dim=1).clamp(min=1)
    cur_len = ys.size(1) - 1

    if step >= 2:
        eos_logit = _eos_logit_behavior0(model, decoder_out, eos_id)
        already = (ys == eos_id).any(dim=1)
        ratio = (cur_len / src_len.float()).clamp(max=1.5)
        eos_logit = eos_logit + 8.0 * ratio
        logits[:, eos_id] = torch.where(already, logits[:, eos_id], eos_logit)

    force = (cur_len >= src_len) & (cur_len >= 2)
    if force.any():
        fill = torch.finfo(logits.dtype).min
        logits[force] = fill
        logits[force, eos_id] = 0.0
    return logits

def optimized_greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    """单序列贪婪解码（仅用于测试）"""
    device = src.device
    memory = model.encode(src, src_mask)
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)
    # 记录生成的行为序列（双头模式）
    generated_behaviors = []
    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device)
        decoder_out, behavior_logits = model.decode(ys, memory, tgt_mask)
        if behavior_logits is not None:
            # 双头：用 behavior_head 预测条件化 item logits
            logits = model.compute_item_logits_from_decoder_out(
                decoder_out, behavior_logits
            )[:, -1]  # [1, V]
            predicted_beh = behavior_logits[:, -1].argmax(-1).item()
            generated_behaviors.append(predicted_beh)
        else:
            logits = decoder_out[:, -1] @ model.item_embedding_decoder.weight.T
        prob = _single_inference_mask_generative(logits, ys) if i >= 2 else _single_inference_mask(logits, src, ys)
        if behavior_logits is not None:
            prob = _apply_eos_with_length_stop(
                prob, model, decoder_out, end_symbol, ys, src, pad=0, step=i
            )
        next_word = prob.argmax(dim=-1).item()
        ys = torch.cat([ys, torch.full((1, 1), next_word, dtype=torch.long, device=device)], dim=1)
        if next_word == end_symbol:
            break
    return ys, generated_behaviors

def optimized_translate(model, src, max_len=50):
    """单序列翻译（仅用于测试）"""
    model.eval()
    src = src.reshape(1, -1)
    src_mask = torch.zeros(src.shape[1], src.shape[1], dtype=torch.bool, device=src.device)
    with torch.no_grad():
        tgt_tokens, generated_behaviors = optimized_greedy_decode(
            model, src, src_mask, max_len=max_len,
            start_symbol=model.SOS, end_symbol=model.EOS
        )
    return tgt_tokens.flatten(), generated_behaviors


# ──────────────────────────────────────────────
#  批量推理核心函数
# ──────────────────────────────────────────────
def batch_encode(model, src, src_key_padding_mask, device):
    """
    批量编码：src [B, S]，利用 src_key_padding_mask 屏蔽 padding 位置。
    复制 RelationGenerator.encode 的逻辑，额外传入 key_padding_mask。
    """
    S = src.size(1)
    position_ids = torch.arange(S, dtype=torch.long, device=device).reshape(1, -1)
    src_item_emb = model.item_embedding(src) + model.position_embedding(position_ids)
    src_emb = model.dropout(src_item_emb)          # eval 模式下 dropout 为恒等
    src_mask = torch.zeros(S, S, dtype=torch.bool, device=device)   # 编码器全注意力
    return model.transformer.encoder(src_emb, src_mask,
                                     src_key_padding_mask=src_key_padding_mask)


def batch_inference_mask(logits, src, ys, PAD=0):
    """
    批量推理掩码（前2步）：只允许 src 中出现过、且尚未生成的 token。
    logits [B, V], src [B, S], ys [B, T]，全向量化，无 Python 循环。
    """
    B, V = logits.shape
    # 初始掩码：允许 src 中的非 PAD token
    mask = torch.zeros(B, V, dtype=torch.bool, device=logits.device)
    safe_src = src.clamp(0, V - 1)
    mask.scatter_(1, safe_src, (src != PAD))          # 非PAD位置设True
    mask[:, PAD] = False                              # PAD 永远不预测
    # 排除已生成的 token（ys）
    safe_ys = ys.clamp(0, V - 1)
    mask.scatter_(1, safe_ys,
                  torch.zeros(B, ys.size(1), dtype=torch.bool, device=logits.device))
    return logits.masked_fill(~mask, float('-inf'))


def batch_inference_mask_generative(logits, ys, PAD=0):
    """
    批量生成掩码（2步后）：允许任何尚未生成过的 token。
    logits [B, V], ys [B, T]，全向量化，无 Python 循环。
    """
    B, V = logits.shape
    mask = torch.ones(B, V, dtype=torch.bool, device=logits.device)
    mask[:, PAD] = False                              # PAD 永远不预测
    safe_ys = ys.clamp(0, V - 1)
    mask.scatter_(1, safe_ys,
                  torch.zeros(B, ys.size(1), dtype=torch.bool, device=logits.device))
    return logits.masked_fill(~mask, float('-inf'))


def batch_greedy_decode(model, memory, src_batch, max_len, SOS, EOS, PAD=0):
    """
    批量贪婪解码（支持双头：同时生成 item 和 behavior 序列）。
    memory   : [B, S, D]  已由 batch_encode 计算好的编码表示
    src_batch: [B, S]     原始（padding后的）源序列，用于 inference_mask
    返回     : (ys [B, T], behaviors [B, T-1] or None)
              behaviors[t] 是生成第 t+1 个 item 时的 behavior_head 预测
    """
    B = src_batch.size(0)
    device = src_batch.device
    use_behavior_head = getattr(model, 'use_behavior_head', False)

    ys = torch.full((B, 1), SOS, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    # 收集每步的 behavior 预测
    all_behaviors = [] if use_behavior_head else None

    for step in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device)
        decoder_out, behavior_logits = model.decode(ys, memory, tgt_mask)

        if use_behavior_head and behavior_logits is not None:
            # 双头：用 behavior_head 预测条件化 item logits
            logits = model.compute_item_logits_from_decoder_out(
                decoder_out, behavior_logits
            )  # [B, T, V]
            logits = logits[:, -1]  # [B, V]
            # 记录本步的 behavior 预测
            step_beh = behavior_logits[:, -1].argmax(-1)  # [B]
            all_behaviors.append(step_beh)
        else:
            logits = decoder_out[:, -1] @ model.item_embedding_decoder.weight.T  # [B, V]

        if step < 2:
            logits = batch_inference_mask(logits, src_batch, ys, PAD)
        else:
            logits = batch_inference_mask_generative(logits, ys, PAD)

        # 前两步只从源复制；之后 EOS 用行为 0，并在达到原序列长度时强制结束
        if use_behavior_head:
            logits = _apply_eos_with_length_stop(
                logits, model, decoder_out, EOS, ys, src_batch, pad=PAD, step=step
            )

        # 兜底：若整行全为 -inf（所有token都被mask），强制输出 EOS
        all_masked = logits.isinf().all(dim=-1)
        if all_masked.any():
            logits[all_masked, EOS] = 0.0

        next_words = logits.argmax(dim=-1)                                # [B]
        next_words = next_words.masked_fill(done, PAD)                   # 已完成→PAD

        ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
        done = done | (next_words == EOS)
        if done.all():
            break

    # 拼接 behavior 序列 [B, num_steps]
    behaviors_tensor = None
    if all_behaviors is not None and len(all_behaviors) > 0:
        behaviors_tensor = torch.stack(all_behaviors, dim=1)  # [B, num_steps]

    return ys, behaviors_tensor

class SequenceRegenerator:
    def __init__(self, config, device, checkpoint_path=None):
        self.config = config
        self.device = device
        self.model = self._load_trained_model(checkpoint_path)
        
        # 特殊token
        self.SOS = self.model.SOS
        self.EOS = self.model.EOS
        self.PAD = self.model.PAD
        
        # 预计算常用tensor
        self._prepare_common_tensors()
        
        logger.info(f"再生器初始化完成：SOS={self.SOS}, EOS={self.EOS}, PAD={self.PAD}")

    def _prepare_common_tensors(self):
        """预计算常用tensor以减少重复创建"""
        self.vocab_size = self.model.item_embedding.num_embeddings
        self.max_seq_len = self.config.training.max_seq_len - 2

    def _load_trained_model(self, checkpoint_path):
        """加载训练好的模型"""
        logger.info("加载预训练再生器...")
        
        if checkpoint_path:
            model_path = Path(checkpoint_path)
        else:
            model_path = Path(self.config.paths.save_dir) / "regenerator_best.pth"
            
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"加载模型权重: {model_path}")
        
        # 加载item2idx
        item2idx_path = Path(self.config.paths.original_sequences).parent / "item2idx.pth"
        item2idx = torch.load(item2idx_path, weights_only=False)
        actual_items = [asin for asin, idx in item2idx.items() if asin != '<PAD>']
        num_items = len(actual_items)
        
        # 初始化模型
        model = RelationGenerator(
            config=self.config,
            num_items=num_items,
            sasrec_emb_path=self.config.paths.sasrec_emb_path
        ).to(self.device)
        
        # 加载权重
        # weights_only=False：检查点里除权重还存了 OmegaConf 的 DictConfig/ListConfig，
        # PyTorch 2.6+ 默认 weights_only=True 会拒绝反序列化这些对象。
        # 这里加载的是自己训练产出的可信检查点，显式关闭安全限制即可。
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        logger.info("再生器加载完成")
        return model

    def generate_single_fast(self, src_seq, condition_idx=0):
        """快速单序列生成"""
        # 快速预处理
        if len(src_seq) > self.max_seq_len:
            src_seq = src_seq[-self.max_seq_len:] 
        
        # 快速验证
        if not all(0 <= t < self.vocab_size for t in src_seq):
            return None
        
        try:
            # 设置条件
            if hasattr(self.model, 'set_condition'):
                self.model.set_condition(condition_idx)
            
            # 生成
            src_tensor = torch.tensor(src_seq, device=self.device)
            generated_seq, generated_behaviors = optimized_translate(self.model, src_tensor, max_len=50)
            generated = generated_seq.cpu().tolist()
            
            # 快速清理
            cleaned_seq = []
            for token in generated:
                if token == self.SOS:
                    continue
                elif token == self.EOS:
                    break
                elif token != self.PAD:
                    cleaned_seq.append(token)
            
            if not cleaned_seq:
                return None

            result = {'generated': cleaned_seq}
            if generated_behaviors:
                # behavior 序列与 item 序列对齐（去掉 SOS 对应的步）
                result['generated_behaviors'] = generated_behaviors[:len(cleaned_seq)]
            return result
            
        except Exception as e:
            logger.error(f"单序列生成异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_all_optimized(self, sequences, K=None, chunk_size=1000, batch_size=64, source_behaviors=None):
        """
        批量并行生成。
        实现要点：
          1. 每个 batch 只编码一次（encode），K 个条件共用 memory，节省 K-1 次 encode。
          2. 同一 batch 内所有序列同步自回归解码（GPU 完全并行）。
          3. 所有 mask 操作向量化，无 Python 逐样本循环。

        source_behaviors: 仅在双头关闭的消融模式下使用；按生成长度从源序列拷贝
        行为标签，使再生数据仍是合法 (item, behavior) 序列。双头模式下传 None。
        """
        if K is None:
            K = self.config.model.K

        all_generated = []
        total_sequences = len(sequences)
        num_batches = (total_sequences + batch_size - 1) // batch_size

        qf = self._quality_filter_cfg()
        logger.info(
            f"开始批量生成：{total_sequences} 个序列，{K} 个条件，"
            f"batch_size={batch_size}，共 {num_batches} 个批次"
        )
        logger.info(
            "质量过滤: "
            f"similarity_threshold={float(qf.get('similarity_threshold', 0.0))}, "
            f"overlap_threshold={float(qf.get('overlap_threshold', 0.0))}, "
            f"min_length={int(qf.get('min_length', 2))}, "
            f"max_length_ratio={float(qf.get('max_length_ratio', 2.0))}"
        )

        self.model.eval()
        t0 = time.time()

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="批量生成", unit="batch", dynamic_ncols=True):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_sequences)
                batch_seqs = sequences[batch_start:batch_end]
                B = len(batch_seqs)

                # ── 1. Pad 源序列 ──────────────────────────────────────────
                max_src_len = max(len(s) for s in batch_seqs)
                src_padded = torch.zeros(B, max_src_len, dtype=torch.long, device=self.device)
                for i, seq in enumerate(batch_seqs):
                    src_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                src_key_padding_mask = (src_padded == self.PAD)   # [B, S]

                # ── 2. 编码一次，所有条件共用 ─────────────────────────────
                try:
                    memory = batch_encode(self.model, src_padded, src_key_padding_mask, self.device)
                except Exception as e:
                    logger.debug(f"Batch {batch_idx} encode 失败: {e}")
                    continue

                # ── 3. 每个条件单独解码 ───────────────────────────────────
                for condition_idx in range(K):
                    if hasattr(self.model, 'set_condition'):
                        self.model.set_condition(condition_idx)

                    try:
                        generated_batch, behavior_batch = batch_greedy_decode(
                            self.model, memory, src_padded,
                            max_len=self.max_seq_len,
                            SOS=self.SOS, EOS=self.EOS, PAD=self.PAD
                        )   # [B, T], [B, num_steps] or None
                    except Exception as e:
                        logger.debug(f"Batch {batch_idx} 条件 {condition_idx} decode 失败: {e}")
                        continue

                    # ── 4. 后处理：清理 token 并质量过滤 ─────────────────
                    for i in range(B):
                        original_seq = batch_seqs[i]
                        tokens = generated_batch[i].cpu().tolist()

                        cleaned = []
                        beh_start = 1  # 跳过 SOS 步
                        for tok_idx, tok in enumerate(tokens):
                            if tok == self.SOS:
                                continue
                            elif tok == self.EOS:
                                break
                            elif tok != self.PAD:
                                cleaned.append(tok)

                        if cleaned and self._check_quality(original_seq, cleaned):
                            entry = {
                                'original': original_seq,
                                'generated': cleaned,
                                'condition': condition_idx,
                                'quality_score': self._calculate_quality_score(original_seq, cleaned)
                            }
                            # 附加 behavior 序列：
                            # - 双头模式：用 behavior_head 的预测
                            # - 消融模式（关双头）：从源序列按位置拷贝，短则截、长则补 0
                            if behavior_batch is not None:
                                beh_tokens = behavior_batch[i].cpu().tolist()
                                # behavior[t] 对应生成的第 t+1 个 item
                                entry['generated_behaviors'] = beh_tokens[:len(cleaned)]
                            elif source_behaviors is not None:
                                src_beh = source_behaviors[batch_start + i] or []
                                if len(src_beh) >= len(cleaned):
                                    entry['generated_behaviors'] = src_beh[:len(cleaned)]
                                else:
                                    entry['generated_behaviors'] = src_beh + [0] * (len(cleaned) - len(src_beh))
                            all_generated.append(entry)

                # ── 5. 定期日志 ────────────────────────────────────────────
                if (batch_idx + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    seqs_done = batch_end
                    speed = seqs_done / elapsed
                    remaining = (total_sequences - seqs_done) / max(speed, 1e-6)
                    logger.info(
                        f"进度 {seqs_done}/{total_sequences} 序列"
                        f" | 已生成 {len(all_generated)} 个变体"
                        f" | {speed:.1f} seq/s"
                        f" | 预计剩余 {remaining/60:.1f} min"
                    )

        elapsed = time.time() - t0
        logger.info(
            f"批量生成完成：{len(all_generated)} 个再生序列"
            f"（{elapsed:.1f}s，平均 {elapsed/max(total_sequences,1)*1000:.1f} ms/seq）"
        )
        return all_generated

    def _quality_filter_cfg(self):
        try:
            return self.config.generation.get("quality_filter", {}) or {}
        except Exception:
            return {}

    def _check_quality(self, original_seq, generated_seq):
        """读取 generation.quality_filter；缺省时退回原先的宽松规则。"""
        if not generated_seq or generated_seq == original_seq:
            return False

        qf = self._quality_filter_cfg()
        min_length = int(qf.get("min_length", 2))
        max_length_ratio = float(qf.get("max_length_ratio", 2.0))
        overlap_threshold = float(qf.get("overlap_threshold", 0.0))
        similarity_threshold = float(qf.get("similarity_threshold", 0.0))

        if len(generated_seq) < min_length:
            return False
        if original_seq and len(generated_seq) > len(original_seq) * max_length_ratio:
            return False

        if overlap_threshold > 0 or similarity_threshold > 0:
            orig_set = set(original_seq)
            gen_set = set(generated_seq)
            union = orig_set | gen_set
            jaccard = (len(orig_set & gen_set) / len(union)) if union else 0.0
            if jaccard < overlap_threshold:
                return False
            if similarity_threshold > 0:
                length_ratio = (
                    min(len(generated_seq), len(original_seq))
                    / max(len(generated_seq), len(original_seq))
                )
                if (jaccard + length_ratio) / 2 < similarity_threshold:
                    return False
        return True

    def _calculate_quality_score(self, original_seq, generated_seq):
        """快速质量分数计算"""
        orig_set = set(original_seq)
        gen_set = set(generated_seq)
        
        overlap = len(orig_set & gen_set)
        union = len(orig_set | gen_set)
        jaccard = overlap / union if union > 0 else 0
        
        length_ratio = min(len(generated_seq), len(original_seq)) / max(len(generated_seq), len(original_seq))
        
        return (jaccard + length_ratio) / 2

@hydra.main(version_base=None, config_path="../configs", config_name="regeneration")
def main(cfg: DictConfig):
    """主函数"""
    seed = cfg.get('seed', 2023)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    device = torch.device(cfg.resources.device)
    logger.info(f"使用设备: {device}")
    
    checkpoint_path = Path(cfg.paths.save_dir) / "regenerator_best.pth"
    
    regenerator = SequenceRegenerator(
        config=cfg,
        device=device,
        checkpoint_path=str(checkpoint_path)
    )
    
    # 加载数据
    item2idx_path = Path(cfg.paths.original_sequences).parent / "item2idx.pth"
    item2idx = torch.load(item2idx_path, weights_only=False)
    logger.info(f"加载item2idx映射: {len(item2idx)}个物品")
    
    sequences_path = Path(cfg.paths.original_sequences)
    sequences_data = torch.load(sequences_path, weights_only=False)
    
    # 处理序列数据
    sequences = []
    # 关双头消融时，从源序列拷贝行为标签，保证再生数据仍是合法 (item, behavior) 序列
    use_behavior_head = bool(getattr(cfg.multi_behavior, "use_behavior_head", True)) \
        if getattr(cfg, "multi_behavior", None) and getattr(cfg.multi_behavior, "enabled", False) else False
    source_behaviors = [] if (not use_behavior_head) else None
    unmapped_count = 0
    
    if isinstance(sequences_data, dict):
        for user_id, user_data in sequences_data.items():
            if isinstance(user_data, dict) and 'sequence' in user_data:
                raw_seq = user_data['sequence']
                mapped_seq = []
                for item in raw_seq:
                    if item in item2idx:
                        mapped_seq.append(item2idx[item])
                    else:
                        unmapped_count += 1
                        continue
                
                if len(mapped_seq) >= 2:
                    sequences.append(mapped_seq)
                    # 消融模式：保留源行为，后续按生成长度对齐拷贝
                    if source_behaviors is not None:
                        raw_beh = user_data.get('behaviors', [])
                        source_behaviors.append([int(b) for b in raw_beh])
    
    logger.info(f"加载了{len(sequences)}个映射后的序列")
    if unmapped_count > 0:
        logger.warning(f"跳过了{unmapped_count}个未在item2idx中的物品")
    
    # 验证第一个序列
    if sequences:
        first_seq = sequences[0]
        logger.info(f"验证第一个序列: {first_seq[:5]}...")
        
        # 快速测试
        test_result = regenerator.generate_single_fast(first_seq, 0)
        if test_result and test_result.get('generated'):
            gen_len = len(test_result['generated'])
            logger.info(f"单序列生成测试成功，生成{gen_len}个token")
            if 'generated_behaviors' in test_result:
                logger.info(f"   行为预测: {test_result['generated_behaviors'][:10]}")
        else:
            logger.error("单序列生成测试失败")
            return
    
    # 批量并行生成
    start_time = time.time()
    regenerated_data = regenerator.generate_all_optimized(
        sequences,
        K=cfg.model.K,
        batch_size=cfg.get('generation_batch_size', 64),  # 可在 config 里覆盖
        source_behaviors=source_behaviors,
    )
    
    generation_time = time.time() - start_time
    logger.info(f"生成耗时: {generation_time:.2f}秒 ({generation_time/60:.1f}分钟)")
    
    # 保存结果
    output_path = Path(cfg.paths.save_data) / "regenerated_data.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'regenerated_pairs': regenerated_data,
        'metadata': {
            'total_pairs': len(regenerated_data),
            'original_sequences': len(sequences),
            'conditions_used': cfg.model.K,
            'generation_time': generation_time,
            'generation_config': cfg
        }
    }, output_path)
    
    logger.info(f"再生数据已保存至: {output_path}")
    logger.info(f"生成统计:")
    logger.info(f"  - 原始序列: {len(sequences)}")
    logger.info(f"  - 再生序列: {len(regenerated_data)}")
    if len(sequences) > 0:
        logger.info(f"  - 平均变体数: {len(regenerated_data) / len(sequences):.2f}")
        logger.info(f"  - 生成速度: {len(sequences) / generation_time:.1f} 序列/秒")

    if regenerated_data:
        gen_lens = np.array([len(p["generated"]) for p in regenerated_data])
        orig_lens = np.array([len(p["original"]) for p in regenerated_data])
        logger.info(
            "  - 再生序列长度: "
            f"最短={int(gen_lens.min())}, "
            f"最长={int(gen_lens.max())}, "
            f"平均={gen_lens.mean():.2f}, "
            f"中位数={int(np.median(gen_lens))}"
        )
        logger.info(
            "  - 对应原序列长度: "
            f"最短={int(orig_lens.min())}, "
            f"最长={int(orig_lens.max())}, "
            f"平均={orig_lens.mean():.2f}, "
            f"中位数={int(np.median(orig_lens))}"
        )

if __name__ == "__main__":
    main()