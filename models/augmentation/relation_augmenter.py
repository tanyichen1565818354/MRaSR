import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor 
import random
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from models.augmentation.graph import GRAPH_FILE_NAMES

logger = logging.getLogger(__name__)


def _qc_dict(qc) -> dict:
    if qc is None:
        return {}
    if isinstance(qc, dict):
        return dict(qc)
    try:
        from omegaconf import OmegaConf
        return dict(OmegaConf.to_container(qc, resolve=True) or {})
    except Exception:
        try:
            return dict(qc)
        except Exception:
            return {}


def quality_control_from_config(config) -> dict:
    try:
        return _qc_dict(config.augmentation.generation.quality_control)
    except Exception:
        return {}


def _seq_overlap(original, augmented):
    orig = [str(x) for x in (original or [])]
    aug = [str(x) for x in (augmented or [])]
    orig_set, aug_set = set(orig), set(aug)
    union = orig_set | aug_set
    jaccard = (len(orig_set & aug_set) / len(union)) if union else 0.0
    novelty = (len(aug_set - orig_set) / len(aug_set)) if aug_set else 0.0
    n_o, n_a = len(orig), len(aug)
    length_ratio = (min(n_o, n_a) / max(n_o, n_a)) if n_o and n_a else 0.0
    n = min(n_o, n_a)
    n_changed = sum(orig[i] != aug[i] for i in range(n)) if n else 0
    n_changed += abs(n_o - n_a)
    replace_ratio = (n_changed / max(n_o, n_a)) if max(n_o, n_a) else 1.0
    return orig, aug, jaccard, novelty, length_ratio, replace_ratio


def _protected_change_count(original, augmented, behaviors, protected):
    if not behaviors:
        return 0, 0
    n = min(len(original), len(augmented), len(behaviors))
    changed = prot_changed = 0
    for i in range(n):
        if str(original[i]) != str(augmented[i]):
            changed += 1
            try:
                if int(behaviors[i]) in protected:
                    prot_changed += 1
            except (TypeError, ValueError):
                pass
    return changed, prot_changed


def filter_augmented_pairs(pairs, qc=None, n_original=None):
    """Keep conservative, high-overlap replacements instead of all generated pairs.

    Sequential recommendation needs the original order mostly intact. Defaults
    (overridable via existing ``quality_control``):
      keep_ratio=0.3, max_per_user=1, max_ratio_to_original=0.3.
    """
    qc = _qc_dict(qc)
    keep_ratio = float(qc.get("keep_ratio", 0.3))
    max_per_user = int(qc.get("max_per_user", 1))
    min_quality = float(qc.get("keep_min_quality", qc.get("quality_threshold", 0.4)))
    min_len = int(qc.get("min_length", 2))
    min_jaccard = float(qc.get("min_jaccard", 0.70))
    max_jaccard = float(qc.get("max_jaccard", 0.995))
    max_replace = float(qc.get("max_replace_ratio", 0.20))
    max_ratio = qc.get("max_ratio_to_original", 0.3)
    protected = {int(x) for x in qc.get("protected_behaviors", (1, 5))}

    if not pairs:
        return [], {"n_input": 0, "n_kept": 0, "keep_ratio": keep_ratio, "reject": {}}

    sample_strategies = {
        p.get("strategy") for p in pairs[:80] if isinstance(p, dict)
    }
    if "plm" not in sample_strategies and "coseq" not in sample_strategies:
        return list(pairs), {
            "n_input": len(pairs),
            "n_kept": len(pairs),
            "keep_ratio": keep_ratio,
            "reject": {},
            "skipped": "not_relation_aug",
        }

    candidates = []
    reject = defaultdict(int)
    seen_aug = set()

    for pair in pairs:
        if not isinstance(pair, dict):
            reject["bad_format"] += 1
            continue
        original = pair.get("original") or []
        augmented = pair.get("augmented") or pair.get("generated") or []
        if not augmented or augmented == original:
            reject["identical_or_empty"] += 1
            continue
        if len(augmented) < min_len:
            reject["too_short"] += 1
            continue
        stored_q = float(pair.get("quality_score", 0.0) or 0.0)
        if stored_q < min_quality:
            reject["low_quality"] += 1
            continue

        orig, aug, jaccard, novelty, length_ratio, replace_ratio = _seq_overlap(
            original, augmented
        )
        if jaccard < min_jaccard:
            reject["jaccard_low"] += 1
            continue
        if jaccard > max_jaccard:
            reject["jaccard_high"] += 1
            continue
        if replace_ratio > max_replace:
            reject["too_many_replaces"] += 1
            continue

        behaviors = pair.get("original_behaviors") or pair.get("behaviors") or []
        _, prot_changed = _protected_change_count(
            original, augmented, behaviors, protected
        )
        if prot_changed > 0:
            reject["protected_changed"] += 1
            continue

        # Prefer small, high-overlap edits that keep like/favorite intact.
        score = (
            0.25 * max(0.0, min(1.0, stored_q))
            + 0.35 * jaccard
            + 0.20 * max(0.0, 1.0 - replace_ratio)
            + 0.10 * length_ratio
            + 0.10 * (1.0 - min(1.0, novelty))
        )
        aug_key = tuple(aug)
        if aug_key in seen_aug:
            reject["duplicate"] += 1
            continue
        seen_aug.add(aug_key)
        candidates.append((score, tuple(orig), pair))

    by_user = defaultdict(list)
    for rec in candidates:
        by_user[rec[1]].append(rec)
    after_user = []
    for recs in by_user.values():
        recs.sort(key=lambda r: r[0], reverse=True)
        after_user.extend(recs[:max_per_user])
        reject["per_user_cap"] += max(0, len(recs) - max_per_user)

    after_user.sort(key=lambda r: r[0], reverse=True)
    target = len(after_user)
    target = min(target, int(len(pairs) * keep_ratio))
    if max_ratio is not None and n_original:
        target = min(target, int(n_original * float(max_ratio)))
    target = max(0, target)
    selected = after_user[:target]

    kept = []
    for score, _, pair in selected:
        out = dict(pair)
        out["filter_score"] = float(score)
        kept.append(out)

    stats = {
        "n_input": len(pairs),
        "n_original_train": n_original,
        "n_candidates": len(candidates),
        "n_after_per_user": len(after_user),
        "n_kept": len(kept),
        "keep_ratio": keep_ratio,
        "max_per_user": max_per_user,
        "max_ratio_to_original": max_ratio,
        "reject": dict(reject),
        "score_mean": (sum(r[0] for r in selected) / len(selected)) if selected else 0.0,
    }
    return kept, stats


def maybe_filter_augmented_payload(payload, qc=None, n_original=None):
    """Filter an augmented pth payload in memory unless it was already filtered."""
    if not isinstance(payload, dict) or "pairs" not in payload:
        return payload, None
    meta = payload.get("metadata") or {}
    if meta.get("filter_applied"):
        logger.info("增强数据已做过质量过滤，跳过")
        return payload, None
    kept, stats = filter_augmented_pairs(
        payload["pairs"], qc=qc, n_original=n_original
    )
    if stats.get("skipped"):
        return payload, None
    if not kept:
        logger.warning("质量过滤后为空，回退到未过滤增强数据")
        return payload, None
    logger.info(
        "增强质量过滤: %s -> %s (keep_ratio=%s, reject=%s, score_mean=%.3f)",
        stats["n_input"],
        stats["n_kept"],
        stats["keep_ratio"],
        stats["reject"],
        stats["score_mean"],
    )
    out = dict(payload)
    out["pairs"] = kept
    out_meta = dict(meta)
    out_meta["filter_applied"] = True
    out_meta["filter_stats"] = stats
    out_meta["total_pairs"] = len(kept)
    out_meta["actual_count"] = len(kept)
    out["metadata"] = out_meta
    return out, stats


def _resolve_graph_file_variant(config) -> str:
    """从 config.graph.file_variant 读取图谱版本，默认 static。"""
    try:
        graph_cfg = config.get("graph", {})
        if graph_cfg and graph_cfg.get("file_variant"):
            return str(graph_cfg["file_variant"])
    except Exception:
        pass
    return "static"


class GraphLoader:
    """图谱数据加载器 - 负责加载预构建的图谱数据"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.augmentation.device)
        self.graph_dir = Path(config.data.graph_save_dir)
        self.plm_emb_dir = Path(config.data.train_dir) / "preprocessed/embeddings"
        self.file_variant = _resolve_graph_file_variant(config)
        self.file_names = GRAPH_FILE_NAMES[self.file_variant]
        
    def load_graph_data(self):
        """加载所有图谱数据"""
        logger.info(f"从 {self.graph_dir} 加载图谱数据 (variant={self.file_variant})...")
        
        # 检查文件存在性
        required_keys = ["plm_emb", "explicit_relations", "plm_similarity", "mappings"]
        required_files = [self.file_names[k] for k in required_keys]
        
        for file in required_files:
            if not (self.graph_dir / file).exists():
                raise FileNotFoundError(f"图谱文件不存在: {self.graph_dir / file}")
        
        # 加载PLM嵌入
        plm_emb = torch.load(self.graph_dir / self.file_names["plm_emb"], weights_only=False).to(self.device)
        logger.info(f"✅ PLM嵌入: {plm_emb.shape}")
        
        plm_id2idx_path = self.plm_emb_dir / "item_id2idx.pth"
        if plm_id2idx_path.exists():
            plm_item_id2idx = torch.load(plm_id2idx_path, weights_only=False)
            logger.info(f"✅ PLM索引映射: {len(plm_item_id2idx)}个物品")
        else:
            logger.warning(f"PLM索引映射文件不存在: {plm_id2idx_path}")
            plm_item_id2idx = {}
        
        # 加载显式关系
        explicit_relations = torch.load(self.graph_dir / self.file_names["explicit_relations"], weights_only=False)
        for k, v in explicit_relations.items():
            explicit_relations[k] = v.to(self.device)
        logger.info(f"✅ 显式关系: {list(explicit_relations.keys())}")
        
        # 加载PLM相似度矩阵
        plm_sim_raw = torch.load(self.graph_dir / self.file_names["plm_similarity"], weights_only=False)
        if not plm_sim_raw.is_coalesced():
            plm_sim_raw = plm_sim_raw.coalesce()
        
        plm_sim_matrix = SparseTensor(
            row=plm_sim_raw.indices()[0],
            col=plm_sim_raw.indices()[1], 
            value=plm_sim_raw.values(),
            sparse_sizes=plm_sim_raw.size()
        ).to(self.device)
        logger.info(f"✅ PLM相似度矩阵: {plm_sim_matrix.nnz()}个非零元素")
        
        # 加载映射关系
        mappings = torch.load(self.graph_dir / self.file_names["mappings"], weights_only=False)
        
        logger.info(f"✅ 映射关系: {len(mappings['item2idx'])}个物品")

        behavior_relation_matrix = None
        matrix_name = self.file_names.get("behavior_matrix")
        if matrix_name:
            matrix_path = self.graph_dir / matrix_name
            if matrix_path.exists():
                matrix_data = torch.load(matrix_path, map_location="cpu", weights_only=False)
                if isinstance(matrix_data, dict):
                    behavior_relation_matrix = matrix_data.get("relation_logits")
                else:
                    behavior_relation_matrix = matrix_data
                logger.info(f"✅ 行为转移矩阵: {matrix_path}")
        
        return {
            'plm_emb': plm_emb,
            'explicit_relations': explicit_relations,
            'plm_sim_matrix': plm_sim_matrix,
            'mappings': mappings,
            'plm_item_id2idx': plm_item_id2idx,
            'behavior_relation_matrix': behavior_relation_matrix,
            'graph_variant': self.file_variant,
        }

class PLMStrategy:
    """基于PLM相似度的增强策略"""
    
    def __init__(self, plm_sim_matrix, item2idx, idx2item, config, device, plm_item_id2idx=None):
        self.plm_sim_matrix = plm_sim_matrix
        self.item2idx = item2idx
        self.idx2item = idx2item
        self.config = config
        self.device = device
        self.plm_item_id2idx = plm_item_id2idx or {}

        # 配置参数
        self.replace_prob = getattr(config, 'replace_prob', 0.3)
        self.similarity_thresh = getattr(config, 'similarity_thresh', 0.5)

        # plm_idx → item_id 反向映射
        self.plm_idx2item = {v: k for k, v in self.plm_item_id2idx.items()}

        logger.info(f"PLMStrategy初始化: PLM映射包含{len(self.plm_item_id2idx)}个物品")
        logger.info(f"PLM相似度矩阵形状: {plm_sim_matrix.sparse_sizes()}")
        if self.plm_item_id2idx:
            plm_indices = list(self.plm_item_id2idx.values())
            logger.info(f"PLM索引范围: {min(plm_indices)} - {max(plm_indices)}")

        # ── 预构建邻居缓存 ──────────────────────────────────────────
        # 把稀疏矩阵的 6.7M 条边一次性转成 {plm_idx: [neighbor_item_id, ...]} 字典
        # 之后每次 _get_similar_item 的查询从 O(6.7M) 降到 O(1)
        logger.info("预构建 PLM 邻居缓存（一次性）...")
        self.neighbor_cache: dict[int, list[str]] = {}
        if self.plm_idx2item:
            row = plm_sim_matrix.storage.row().cpu()    # [nnz]
            col = plm_sim_matrix.storage.col().cpu()    # [nnz]
            val = plm_sim_matrix.storage.value().cpu()  # [nnz]
            thresh = self.similarity_thresh

            for r, c, v in zip(row.tolist(), col.tolist(), val.tolist()):
                if v < thresh:
                    continue
                neighbor_item = self.plm_idx2item.get(c)
                if neighbor_item is None:
                    continue
                if r not in self.neighbor_cache:
                    self.neighbor_cache[r] = []
                self.neighbor_cache[r].append(neighbor_item)

        logger.info(f"邻居缓存构建完成: {len(self.neighbor_cache)} 个物品有邻居")
        
    def generate(self, original_seq, behavior_replace_probs=None):
        """
        在PLM替换基础上，插入/删除操作也用PLM语义相关性。
        behavior_replace_probs: 每个位置的替换概率列表（多行为模式下传入），
                                None 时使用策略统一的 self.replace_prob。
        """
        if not original_seq:
            return original_seq

        seq = list(original_seq)
        plm_emb = getattr(self, 'plm_emb', None)
        plm_item_id2idx = self.plm_item_id2idx

        # 插入操作（20% 概率）——从邻居缓存取候选，O(1) 而非全量扫描
        seq_set = set(seq)
        if random.random() < 0.2 and len(seq) < 50 and len(seq) > 2:
            insert_pos = random.randint(1, len(seq) - 1)
            left, right = seq[insert_pos - 1], seq[insert_pos]
            # 合并左右邻居，排除已在序列中的物品
            left_nb  = self._get_similar_item_full(left)
            right_nb = self._get_similar_item_full(right)
            candidates = [it for it in set(left_nb + right_nb) if it not in seq_set]
            if candidates:
                insert_item = random.choice(candidates)
                seq = seq[:insert_pos] + [insert_item] + seq[insert_pos:]
                seq_set.add(insert_item)

        # 删除操作（15% 概率）——删除与上下文相似度最低的物品
        # 向量化：一次性算出所有候选的相似度，避免逐个 CUDA kernel launch
        elif random.random() < 0.15 and len(seq) > 3 and plm_emb is not None and plm_item_id2idx:
            del_candidates = []
            for i in range(1, len(seq) - 1):
                left, right = seq[i - 1], seq[i + 1]
                if left in plm_item_id2idx and right in plm_item_id2idx and seq[i] in plm_item_id2idx:
                    del_candidates.append(i)
            if del_candidates:
                # 批量构造 context_vec 和 item_vec，一次 cosine_similarity
                ctx_left = plm_emb[torch.tensor([plm_item_id2idx[seq[i - 1]] for i in del_candidates], device=plm_emb.device)]
                ctx_right = plm_emb[torch.tensor([plm_item_id2idx[seq[i + 1]] for i in del_candidates], device=plm_emb.device)]
                item_vecs = plm_emb[torch.tensor([plm_item_id2idx[seq[i]] for i in del_candidates], device=plm_emb.device)]
                ctx_vec = (ctx_left + ctx_right) / 2.0
                sims = F.cosine_similarity(item_vecs, ctx_vec, dim=1)
                min_sim, del_offset = torch.min(sims, dim=0)
                if min_sim.item() < 0.7:
                    del_idx = del_candidates[del_offset.item()]
                    seq = seq[:del_idx] + seq[del_idx + 1:]

        # PLM替换策略（支持行为感知的逐位替换概率）
        new_seq = []
        for pos, item in enumerate(seq):
            # 优先使用行为感知概率，否则使用统一概率
            if behavior_replace_probs is not None and pos < len(behavior_replace_probs):
                cur_replace_prob = behavior_replace_probs[pos]
            else:
                cur_replace_prob = self.replace_prob

            if random.random() < cur_replace_prob:
                replacement = self._get_similar_item(item)
                if replacement and replacement != item:
                    new_seq.append(replacement)
                else:
                    new_seq.append(item)
            else:
                new_seq.append(item)

        if len(new_seq) > 47:
            new_seq = new_seq[:47]
        return [item for item in new_seq if item != '<PAD>']
    
    def _get_similar_item(self, item):
        """从预构建缓存中 O(1) 随机返回一个相似物品"""
        plm_idx = self.plm_item_id2idx.get(item)
        if plm_idx is None:
            return None
        neighbors = self.neighbor_cache.get(plm_idx)
        if not neighbors:
            return None
        return random.choice(neighbors)

    def _get_similar_item_full(self, item):
        """（仅供插入操作使用）返回多个相似物品列表"""
        plm_idx = self.plm_item_id2idx.get(item)
        if plm_idx is None:
            return []
        return self.neighbor_cache.get(plm_idx, [])

    def _get_similar_item_legacy(self, item):
        """（保留备用，正常路径不走这里）"""
        if item not in self.item2idx:
            return None

        try:
            item_idx = self.item2idx[item]
            if item_idx == 0:
                return None

            plm_idx = self.plm_item_id2idx.get(item, item_idx - 1)
            if plm_idx < 0 or plm_idx >= self.plm_sim_matrix.sparse_sizes()[0]:
                return None

            indices = self.plm_sim_matrix.storage.row()
            col_indices = self.plm_sim_matrix.storage.col()
            values = self.plm_sim_matrix.storage.value()
            row_mask = indices == plm_idx
            if not row_mask.any():
                return None
            candidates_indices = col_indices[row_mask]
            candidates_scores = values[row_mask]
            thresh_mask = candidates_scores > self.similarity_thresh
            candidates_indices = candidates_indices[thresh_mask]
            candidates_scores = candidates_scores[thresh_mask]
            if len(candidates_indices) == 0:
                return None

            # 多次采样避免选择同一个商品
            max_attempts = 3  # 减少尝试次数
            used_items = set()
            
            for attempt in range(max_attempts):
                selected_idx = torch.multinomial(probs, 1).item()
                selected_plm_idx = candidates_indices[selected_idx].item()
                
                # 将PLM索引转回商品ID
                if selected_plm_idx in plm_idx_to_items:
                    candidates = [candidate_item for candidate_item in plm_idx_to_items[selected_plm_idx] 
                                if candidate_item != item and candidate_item not in used_items]
                    if candidates:
                        selected_item = random.choice(candidates)
                        used_items.add(selected_item)
                        return selected_item
                else:
                    logger.debug(f"PLM索引 {selected_plm_idx} 不在映射中")
            
            # 备用选择逻辑
            for i, candidate_plm_idx in enumerate(candidates_indices.cpu().numpy()):
                if candidate_plm_idx in plm_idx_to_items:
                    candidates = [candidate_item for candidate_item in plm_idx_to_items[candidate_plm_idx] 
                                if candidate_item != item and candidate_item not in used_items]
                    if candidates:
                        selected_item = random.choice(candidates)
                        return selected_item
                else:
                    logger.debug(f"备用候选PLM索引 {candidate_plm_idx} 不在映射中")
        
        except Exception as e:
            logger.debug(f"获取相似物品失败 {item}: {str(e)}")
            return None
        
        return None

class ExplicitRelationStrategy:
    """基于显式关系的增强策略"""
    
    def __init__(self, explicit_relations, item2idx, idx2item, config):
        self.co_matrix = explicit_relations['co_occurrence']
        self.seq_matrix = explicit_relations['sequential']
        self.item2idx = item2idx
        self.idx2item = idx2item
        self.config = config.augmentation.generation.strategies.coseq

        # ── 预构建邻居缓存（一次性）──────────────────────────────────
        # 把稀疏矩阵的每条边转成 {idx: {neighbor_idx: score}} 字典，
        # 之后每次查询从 O(nnz) 全量扫描降到 O(1)。
        # 不做这步的话，15 万条序列 × 每条 ~120 次稀疏矩阵扫描 = 25 小时。
        logger.info("预构建显式关系邻居缓存（co_occurrence + sequential）...")
        self.co_neighbors = self._build_neighbor_cache(self.co_matrix)
        self.seq_neighbors = self._build_neighbor_cache(self.seq_matrix)
        logger.info(f"邻居缓存构建完成: co={len(self.co_neighbors)}, seq={len(self.seq_neighbors)} 个物品有邻居")

    def _build_neighbor_cache(self, matrix):
        """把稀疏矩阵转成 {row_idx: {col_idx: value}} 的字典缓存。"""
        cache = defaultdict(dict)
        matrix = matrix.coalesce()
        rows = matrix.indices()[0].cpu().tolist()
        cols = matrix.indices()[1].cpu().tolist()
        vals = matrix.values().cpu().tolist()
        for r, c, v in zip(rows, cols, vals):
            if c != 0:  # 排除 PAD
                cache[r][c] = v
        return dict(cache)
        
    def generate(self, original_seq, behavior_replace_probs=None):
        """
        在显式关系加权基础上，插入/删除/替换操作都用cooccur+sequential分数。
        behavior_replace_probs: 每个位置的替换概率列表（多行为模式下传入），
                                None 时使用策略统一概率。
        """
        if not original_seq:
            return original_seq

        seq = list(original_seq)
        insert_prob = 0.2
        delete_prob = 0.15
        replace_prob = self.config.replace_prob if hasattr(self.config, 'replace_prob') else 0.15
        topk = 5
        min_del_score = 0.1

        # 插入操作（20%概率）
        if random.random() < insert_prob and len(seq) < 50:
            if len(seq) > 2:
                insert_pos = random.randint(1, len(seq)-1)
                left = seq[insert_pos-1]
                right = seq[insert_pos]
                
                # 获取left和right的候选及分数
                left_scores = self._get_all_relation_scores(left)
                right_scores = self._get_all_relation_scores(right)
                
                # 统计所有候选的平均分数
                candidate_scores = defaultdict(float)
                candidate_count = defaultdict(int)
                for item, score in left_scores.items():
                    candidate_scores[item] += score
                    candidate_count[item] += 1
                for item, score in right_scores.items():
                    candidate_scores[item] += score
                    candidate_count[item] += 1
                # 取平均
                for item in candidate_scores:
                    candidate_scores[item] /= candidate_count[item]
                # 排除已在序列中的item和None
                valid_candidates = [item for item in candidate_scores if item not in seq and item is not None]
                if valid_candidates:
                    # 选topk分数最高的，再随机采样
                    sorted_candidates = sorted(valid_candidates, key=lambda x: candidate_scores[x], reverse=True)
                    top_candidates = sorted_candidates[:min(topk, len(sorted_candidates))]
                    insert_item = random.choice(top_candidates)
                    seq = seq[:insert_pos] + [insert_item] + seq[insert_pos:]

        # 删除操作（15%概率）
        elif random.random() < delete_prob and len(seq) > 3:
            del_candidates = [i for i in range(1, len(seq)-1)]
            min_score = float('inf')
            del_idx = None
            for i in del_candidates:
                left = seq[i-1]
                right = seq[i+1]
                item = seq[i]
                # 计算item与前后item的平均加权分数
                left_score = self._get_relation_score(item, left)
                right_score = self._get_relation_score(item, right)
                avg_score = (left_score + right_score) / 2
                if avg_score < min_score:
                    min_score = avg_score
                    del_idx = i
            if del_idx is not None and min_score < min_del_score:
                seq = seq[:del_idx] + seq[del_idx+1:]

        # 替换操作：支持行为感知的逐位替换概率
        core_size = max(1, len(seq) // 3)
        new_seq = seq[:core_size].copy()
        for i in range(core_size, len(seq)):
            # 行为感知替换概率
            if behavior_replace_probs is not None and i < len(behavior_replace_probs):
                cur_prob = behavior_replace_probs[i]
            else:
                cur_prob = replace_prob

            if random.random() < cur_prob:
                replacement = self._get_related_item(seq[i])
                new_seq.append(replacement if replacement else seq[i])
            else:
                new_seq.append(seq[i])

        if len(new_seq) > 47:
            new_seq = new_seq[:47]
        return [item for item in new_seq if item != '<PAD>']

    def _get_all_relation_scores(self, item):
        """获取item与所有物品的加权融合分数（共现+时序）"""
        if item not in self.item2idx:
            return {}
        idx = self.item2idx[item]
        if idx == 0:
            return {}
        idx = idx - 1
        scores = defaultdict(float)
        # 共现
        co = self._get_sparse_candidates(self.co_matrix, idx)
        for k, v in co.items():
            original_idx = k + 1
            candidate_item = self.idx2item.get(original_idx)
            if candidate_item and candidate_item != '<PAD>' and candidate_item != item:
                scores[candidate_item] += v * self.config.cooccur_weight
        # 时序
        seq = self._get_sparse_candidates(self.seq_matrix, idx)
        for k, v in seq.items():
            original_idx = k + 1
            candidate_item = self.idx2item.get(original_idx)
            if candidate_item and candidate_item != '<PAD>' and candidate_item != item:
                scores[candidate_item] += v * self.config.sequential_weight
        return scores

    def _get_relation_score(self, item1, item2):
        """获取item1与item2的加权融合分数"""
        if item1 not in self.item2idx or item2 not in self.item2idx:
            return 0.0
        idx1 = self.item2idx[item1] - 1
        idx2 = self.item2idx[item2] - 1
        score = 0.0
        # 共现 + 时序：从缓存 O(1) 查询，不再重复扫描稀疏矩阵
        co_neighbors = self.co_neighbors.get(idx1, {})
        if idx2 in co_neighbors:
            score += co_neighbors[idx2] * self.config.cooccur_weight
        seq_neighbors = self.seq_neighbors.get(idx1, {})
        if idx2 in seq_neighbors:
            score += seq_neighbors[idx2] * self.config.sequential_weight
        return score
    
    def _get_related_item(self, item):
        """获取相关物品"""
        if item not in self.item2idx:
            return None
            
        idx = self.item2idx[item]
        if idx == 0:  # 如果是PAD，直接返回None
            return None
        
        # 因为矩阵不包含PAD，所以需要减1
        idx = idx - 1
        
        try:
            candidates = defaultdict(float)
            
            # 从共现关系获取候选
            co_candidates = self._get_sparse_candidates(self.co_matrix, idx)
            for candidate_idx, weight in co_candidates.items():
                original_idx = candidate_idx + 1
                candidate_item = self.idx2item.get(original_idx)
                if candidate_item and candidate_item != '<PAD>' and candidate_item != item:
                    candidates[candidate_item] += weight * self.config.cooccur_weight
            
            # 从时序关系获取候选
            seq_candidates = self._get_sparse_candidates(self.seq_matrix, idx)
            for candidate_idx, weight in seq_candidates.items():
                original_idx = candidate_idx + 1
                candidate_item = self.idx2item.get(original_idx)
                if candidate_item and candidate_item != '<PAD>' and candidate_item != item:
                    candidates[candidate_item] += weight * self.config.sequential_weight
            
            # 过滤低权重候选
            valid_candidates = {
                k: v for k, v in candidates.items() 
                if v > self.config.min_threshold
            }
            
            if not valid_candidates:
                return None
            
            # 按权重随机选择
            items = list(valid_candidates.keys())
            weights = list(valid_candidates.values())
            return random.choices(items, weights=weights, k=1)[0]
            
        except Exception as e:
            logger.warning(f"获取相关物品失败 {item}({idx}): {e}")
            return None
    
    def _get_sparse_candidates(self, matrix, idx):
        """从邻居缓存 O(1) 获取候选物品（兼容旧接口，matrix 参数仅用于区分 co/seq）"""
        if matrix is self.co_matrix:
            return dict(self.co_neighbors.get(idx, {}))
        elif matrix is self.seq_matrix:
            return dict(self.seq_neighbors.get(idx, {}))
        # 兜底：未识别的矩阵走老路径（不应发生）
        candidates = {}
        try:
            matrix = matrix.coalesce()
            row_mask = matrix.indices()[0] == idx
            if row_mask.any():
                cols = matrix.indices()[1][row_mask]
                values = matrix.values()[row_mask]
                for col, val in zip(cols, values):
                    if col.item() != 0:
                        candidates[col.item()] = val.item()
        except Exception as e:
            logger.warning(f"从稀疏矩阵获取候选失败: {e}")
        return candidates

class RelationAugmenter:
    """简洁版关系增强器"""
    
    def __init__(self, products, sequences, config):
        """初始化增强器"""
        self.config = config
        self.device = torch.device(config.augmentation.device)
        self.products = products
        self.sequences = sequences

        # 设置随机种子
        seed = config.get("seed", 2025)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.last_filter_stats = None
        
        # 加载图谱数据
        self.graph_loader = GraphLoader(config)
        self.graph_data = self.graph_loader.load_graph_data()
        
        # 提取映射关系
        self.item2idx = self.graph_data['mappings']['item2idx']
        self.idx2item = self.graph_data['mappings']['idx2item']
        
        # 🔧 获取PLM索引映射
        plm_item_id2idx = self.graph_data.get('plm_item_id2idx', {})
        
        # 初始化增强策略
        self.plm_strategy = PLMStrategy(
            self.graph_data['plm_sim_matrix'],
            self.item2idx,
            self.idx2item,
            config.augmentation.generation.strategies.plm,
            self.device,
            plm_item_id2idx
        )

        self.plm_strategy.plm_emb = self.graph_data['plm_emb']
        
        self.explicit_strategy = ExplicitRelationStrategy(
            self.graph_data['explicit_relations'],
            self.item2idx,
            self.idx2item,
            config
        )
        
        logger.info("✅ 简洁版关系增强器初始化完成")
    
    def generate_augmented_sequences(self):
        """生成增强序列 - 完整版本"""
        logger.info("开始生成增强序列（完整模式）...")
        
        # 启用两种策略
        enabled_strategies = []
        if self.config.augmentation.generation.strategies.plm.enabled:
            enabled_strategies.append('plm')
        if self.config.augmentation.generation.strategies.coseq.enabled:
            enabled_strategies.append('coseq')
        
        logger.info(f"启用策略: {enabled_strategies}")

        # 目标生成数量（必须在使用前赋值）
        target_count = self.config.get('target_count', 50000)
        logger.info(f"目标生成数量: {target_count}")

        # 准备处理序列，过滤过短序列
        sequences_to_process = [
            (seq_id, seq_data) for seq_id, seq_data in self.sequences.items()
            if len(seq_data['sequence']) >= self.config.sequence.min_length
        ]

        # 估算最少需要处理多少序列，避免无谓遍历全量
        variants = self.config.augmentation.generation.variants_per_strategy
        n_strategies = len(enabled_strategies)
        min_needed = max(1, target_count // max(1, variants * n_strategies))
        if len(sequences_to_process) > min_needed * 3:
            # 随机打乱，只保留 3 倍冗余量（考虑部分序列增强失败）
            seed = self.config.get("seed", 2025)
            rng = random.Random(seed)
            rng.shuffle(sequences_to_process)
            sequences_to_process = sequences_to_process[:min_needed * 3]

        logger.info(f"完整模式：处理 {len(sequences_to_process)} 个序列（目标 {target_count} 对）")
        
        # 🔧 修复：强制使用串行处理，避免CUDA线程安全问题
        logger.info("使用串行处理以避免CUDA线程安全问题")
        all_pairs = self._sequential_generate(sequences_to_process, enabled_strategies, target_count)
        
        logger.info(f"✅ 完整生成完成，共生成{len(all_pairs)}对增强序列")

        qc = quality_control_from_config(self.config)
        generated_pairs = all_pairs
        all_pairs, filter_stats = filter_augmented_pairs(
            all_pairs, qc=qc, n_original=len(self.sequences)
        )
        if not all_pairs and generated_pairs:
            logger.warning("质量过滤后为空，回退到未过滤增强数据")
            all_pairs = generated_pairs
        self.last_filter_stats = filter_stats
        logger.info(
            "质量过滤: %s -> %s (keep_ratio=%s, reject=%s, score_mean=%.3f)",
            filter_stats.get("n_input", 0),
            filter_stats.get("n_kept", 0),
            filter_stats.get("keep_ratio"),
            filter_stats.get("reject"),
            filter_stats.get("score_mean", 0.0),
        )

        # 统计长度变化
        orig_lens = []
        aug_lens = []
        for pair in all_pairs:
            orig_lens.append(len(pair['original']))
            aug_lens.append(len(pair['augmented']))
        if orig_lens and aug_lens:
            orig_avg = np.mean(orig_lens)
            orig_min = np.min(orig_lens)
            orig_max = np.max(orig_lens)
            aug_avg = np.mean(aug_lens)
            aug_min = np.min(aug_lens)
            aug_max = np.max(aug_lens)
            diff_avg = aug_avg - orig_avg
            logger.info(f"原始序列长度: 平均={orig_avg:.2f}, 最小={orig_min}, 最大={orig_max}")
            logger.info(f"增强序列长度: 平均={aug_avg:.2f}, 最小={aug_min}, 最大={aug_max}")
            logger.info(f"增强序列长度-原始序列长度: 平均差值={diff_avg:.2f}")

        return self._format_results(all_pairs)
    
    def _sequential_generate(self, sequences_to_process, enabled_strategies, target_count):
        """串行生成增强序列"""
        all_pairs = []
        processed_count = 0
        
        # 🔧 优化：更准确的进度预估
        total_sequences = len(sequences_to_process)
        expected_pairs_per_seq = len(enabled_strategies) * self.config.augmentation.generation.variants_per_strategy
        
        logger.info(f"预期每个序列生成 {expected_pairs_per_seq} 对增强序列")
        
        with tqdm(total=total_sequences, desc="处理序列", 
                  unit="序列", smoothing=0.1) as pbar:
            
            for seq_id, seq_data in sequences_to_process:
                if len(all_pairs) >= target_count:
                    logger.info(f"已达到目标数量 {target_count}，停止处理")
                    break
                    
                try:
                    pairs = self._process_single_sequence(seq_id, seq_data, enabled_strategies)
                    all_pairs.extend(pairs)
                    processed_count += 1
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        "已生成": len(all_pairs),
                        "目标": target_count,
                        "完成率": f"{len(all_pairs)/target_count*100:.1f}%",
                        "当前效率": f"{len(pairs)}/{expected_pairs_per_seq}"
                    })
                    
                    # 定期输出详细进度
                    if processed_count % 2000 == 0:
                        avg_pairs_per_seq = len(all_pairs) / processed_count if processed_count > 0 else 0
                        estimated_total = avg_pairs_per_seq * total_sequences
                        logger.info(f"已处理 {processed_count}/{total_sequences} 个序列")
                        logger.info(f"已生成 {len(all_pairs)} 对增强序列")
                        logger.info(f"平均每序列生成 {avg_pairs_per_seq:.1f} 对")
                        logger.info(f"预计总共可生成 {estimated_total:.0f} 对")

                except Exception as e:
                    logger.debug(f"序列处理失败 {seq_id}: {e}")
                    continue
        
        return all_pairs
    
    def _get_behavior_replace_probs(self, behaviors):
        """
        根据多行为配置返回每个位置的替换概率列表。
        若未启用多行为，或显式关闭行为感知替换（use_behavior_replace_probs=false），
        返回 None（各策略使用统一的 replace_prob）。
        """
        try:
            mb_cfg = self.config.get("multi_behavior", None)
            if (
                mb_cfg
                and mb_cfg.get("enabled", False)
                and mb_cfg.get("use_behavior_replace_probs", True)
            ):
                brp = mb_cfg.get("behavior_replace_probs", {})
                return [float(brp.get(str(b), 0.15)) for b in behaviors]
        except Exception:
            pass
        return None

    def _process_single_sequence(self, seq_id, seq_data, enabled_strategies):
        """处理单个序列（支持多行为感知）"""
        original_seq = seq_data['sequence']
        # 多行为：获取 behaviors 列表，不存在时全部视为 purchase(2)
        original_behaviors = seq_data.get('behaviors', [2] * len(original_seq))

        # 计算每个位置的行为感知替换概率
        behavior_replace_probs = self._get_behavior_replace_probs(original_behaviors)

        results = []
        variants_per_strategy = self.config.augmentation.generation.variants_per_strategy

        for strategy in enabled_strategies:
            success_count = 0
            attempts = 0
            max_attempts = variants_per_strategy * 3

            while success_count < variants_per_strategy and attempts < max_attempts:
                try:
                    attempts += 1

                    if strategy == 'plm':
                        # 将行为感知替换概率注入策略（临时覆盖）
                        augmented = self.plm_strategy.generate(
                            original_seq,
                            behavior_replace_probs=behavior_replace_probs
                        )
                    elif strategy == 'coseq':
                        augmented = self.explicit_strategy.generate(
                            original_seq,
                            behavior_replace_probs=behavior_replace_probs
                        )
                    else:
                        break

                    if (augmented and
                        len(augmented) >= self.config.sequence.min_length and
                        len(augmented) <= self.config.augmentation.generation.max_length and
                        augmented != original_seq):

                        quality_score = self._evaluate_quality(augmented, original_seq)
                        quality_threshold = self.config.augmentation.generation.quality_control.quality_threshold
                        if quality_score > quality_threshold:
                            results.append({
                                'original': original_seq,
                                'augmented': augmented,
                                'original_behaviors': original_behaviors,
                                'strategy': strategy,
                                'user_id': seq_id[:6],
                                'quality_score': quality_score
                            })
                            success_count += 1

                except Exception as e:
                    logger.debug(f"生成失败: {e}")
                    continue

        return results
    
    def _evaluate_quality(self, augmented, original):
        """Prefer conservative replacements that keep most of the original sequence."""
        if not augmented or len(augmented) < 2:
            return 0.0
        _, _, jaccard, novelty, length_ratio, replace_ratio = _seq_overlap(
            original, augmented
        )
        if jaccard <= 0.0:
            return 0.0
        score = (
            0.45 * jaccard
            + 0.25 * max(0.0, 1.0 - replace_ratio)
            + 0.15 * max(0.0, 1.0 - novelty)
            + 0.15 * length_ratio
        )
        return max(0.0, min(1.0, score))
    
    def _format_results(self, pairs):
        """格式化结果（保留 original_behaviors 字段供再生器使用）"""
        formatted_pairs = []
        for pair in pairs:
            formatted_pairs.append({
                'original': pair['original'],
                'augmented': pair['augmented'],
                'original_behaviors': pair.get('original_behaviors', []),
                'strategy': pair['strategy'],
                'user_id': pair['user_id'],
                'quality_score': pair['quality_score']
            })
        return formatted_pairs
    
    def prepare_augment(self):
        """准备增强（兼容接口）"""
        logger.info("图谱数据已加载，准备完成")
        pass