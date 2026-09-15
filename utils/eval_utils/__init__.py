import torch
import numpy as np
from pathlib import Path

def recall_at_k(preds, targets, k=10):
    """DR4SR风格的召回率计算 - 支持多物品评估"""
    # preds: [batch_size, num_items]
    # targets: 可以是单个目标索引 [batch_size] 或多目标掩码 [batch_size, num_items]
    
    _, topk_indices = torch.topk(preds, k, dim=-1)  # [batch_size, k]
    
    if targets.dim() == 1:
        # 单目标情况：将目标转换为掩码
        target_mask = torch.zeros_like(preds, dtype=torch.bool)
        batch_indices = torch.arange(targets.size(0), device=preds.device)
        target_mask[batch_indices, targets] = True
    else:
        # 多目标情况：直接使用目标掩码
        target_mask = targets > 0
    
    # 为topk创建掩码
    topk_mask = torch.zeros_like(preds, dtype=torch.bool)
    batch_indices = torch.arange(preds.size(0), device=preds.device).unsqueeze(1).expand(-1, k)
    topk_mask[batch_indices, topk_indices] = True
    
    # 计算召回率 (预测正确的物品数 / 总的相关物品数)
    hits = (topk_mask & target_mask).sum(dim=1).float()
    num_targets = target_mask.sum(dim=1).float()
    
    # 防止分母为0
    valid_mask = num_targets > 0
    recall = torch.zeros_like(num_targets)
    recall[valid_mask] = hits[valid_mask] / num_targets[valid_mask]
    
    return recall.mean()

def ndcg_at_k(preds, targets, k=10):
    """DR4SR风格的NDCG计算 - 支持多物品评估"""
    # preds: [batch_size, num_items]
    # targets: 可以是单个目标索引 [batch_size] 或多目标掩码 [batch_size, num_items]
    
    _, topk_indices = torch.topk(preds, k, dim=-1)  # [batch_size, k]
    
    if targets.dim() == 1:
        # 单目标情况：将目标转换为掩码
        target_mask = torch.zeros_like(preds, dtype=torch.bool)
        batch_indices = torch.arange(targets.size(0), device=preds.device)
        target_mask[batch_indices, targets] = True
    else:
        # 多目标情况：直接使用目标掩码
        target_mask = targets > 0
    
    # 构建位置折扣
    ranks = torch.arange(1, k+1, device=preds.device).float()
    discounts = 1.0 / torch.log2(ranks + 1)  # [k]
    
    # 计算每个位置的相关性 (1=相关, 0=不相关)
    batch_indices = torch.arange(preds.size(0), device=preds.device).unsqueeze(1).expand(-1, k)
    relevance = target_mask[batch_indices, topk_indices].float()  # [batch_size, k]
    
    # 计算DCG
    dcg = (relevance * discounts).sum(dim=1)  # [batch_size]
    
    # 计算IDCG (理想排序情况下的DCG)
    num_targets = torch.clamp(target_mask.sum(dim=1), max=k)  # 最多考虑k个相关物品
    ideal_relevance = torch.zeros((preds.size(0), k), device=preds.device)
    for i in range(preds.size(0)):
        ideal_relevance[i, :min(int(num_targets[i]), k)] = 1.0
    idcg = (ideal_relevance * discounts).sum(dim=1)  # [batch_size]
    
    # 处理IDCG为0的情况
    valid_mask = idcg > 0
    ndcg = torch.zeros_like(dcg)
    ndcg[valid_mask] = dcg[valid_mask] / idcg[valid_mask]
    
    return ndcg.mean()

def evaluate(model, eval_loader, k=10, device=None, ks=None):
    """向量化的留一评估，一次前向 + 一次 topk(max_k) 同时算多个 k。

    参数：
      k     : 单个 k（兼容旧接口），返回 ``{recall@k, ndcg@k}``。
      ks    : 多个 k 的元组/列表（如 ``(10, 20)``）。若给出则忽略 ``k``，
              返回 ``{recall@k1, ndcg@k1, recall@k2, ndcg@k2, ...}``。

    相比旧实现的关键加速：
      - 历史物品屏蔽用 ``scatter_`` 一次性完成，不再逐样本 Python 循环。
      - topk 只算一次 ``max(ks)``，多个 k 通过掩码派生，不再重复跑前向。
      - 命中位置用向量化的 ``argmax`` 取得，不再逐样本 ``.item()`` 同步。
    """
    model.eval()

    if ks is not None:
        ks = tuple(sorted(ks))
    else:
        ks = (int(k),)
    k_max = max(ks)

    cum_recall = {kk: 0.0 for kk in ks}
    cum_ndcg = {kk: 0.0 for kk in ks}
    total_samples = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {kk: v.to(device) for kk, v in batch.items()}
            query = model(batch, need_pooling=True)
            if query.dim() != 2:
                raise ValueError(
                    f"Expected query to be 2D [batch_size, embed_dim], got {query.shape}"
                )

            # 与所有物品打分
            scores = query @ model.item_embedding.weight.T  # [B, N]

            # 一次性屏蔽历史物品 + padding（scatter_ 对重复 idx 设置常数安全）
            in_items = batch['in_item_id']  # [B, L]，含 0 → 同步屏蔽 padding 列
            scores.scatter_(1, in_items, -1e9)

            targets = batch['item_id']  # [B]，留一目标
            valid = targets > 0
            if not valid.any():
                continue
            scores_v = scores[valid]            # [Bv, N]
            targets_v = targets[valid]          # [Bv]

            # 一次 topk(max_k)，多个 k 都从这派生
            _, topk_idx = torch.topk(scores_v, k=k_max, dim=1)  # [Bv, k_max]
            hits = topk_idx == targets_v.unsqueeze(1)           # [Bv, k_max] bool
            hit_any = hits.any(dim=1)                           # [Bv]
            # 命中位置（0-indexed）；未命中记为 k_max（不影响 < k 的判断）
            ranks = torch.where(
                hit_any,
                hits.float().argmax(dim=1),
                torch.full_like(hits.float().argmax(dim=1), k_max),
            )  # [Bv] long

            n_valid = int(targets_v.size(0))
            total_samples += n_valid

            # 预计算 1/log2(rank+2)，供各 k 复用
            log2_discount = 1.0 / torch.log2(ranks.float() + 2.0)  # [Bv]

            for kk in ks:
                hit_at_k = hit_any & (ranks < kk)  # [Bv]
                cum_recall[kk] += float(hit_at_k.float().sum().item())
                ndcg_vals = torch.where(
                    hit_at_k, log2_discount, torch.zeros_like(log2_discount)
                )
                cum_ndcg[kk] += float(ndcg_vals.sum().item())

    if total_samples == 0:
        out = {}
        for kk in ks:
            out[f'recall@{kk}'] = 0.0
            out[f'ndcg@{kk}'] = 0.0
        return out

    out = {}
    for kk in ks:
        out[f'recall@{kk}'] = cum_recall[kk] / total_samples
        out[f'ndcg@{kk}'] = cum_ndcg[kk] / total_samples
    return out


def evaluate_and_dump(model, eval_loader, dump_path, k=20, device=None, ks=None):
    """Same as `evaluate`, but also dumps per-user predictions and histories
    to a .npz file for later bucketed analysis (paper Fig. 11, Tables 9-11).

    The multi-behavior LazyNextItemDataset provides `in_item_id` and
    `in_behavior_id` in each batch, so both are saved.

    Saved .npz keys:
        scores      : int64 [N, k_max]   top-k_max item ids per user
        labels      : int64 [N]          ground-truth next item per user
        hist_items  : int64 [N, L]       history item ids (padded with 0)
        hist_behs  : int64 [N, L]       history behavior ids (padded with 0)
        user_idx   : int64 [N]          user index in the eval set

    Note: item popularity is not available inside the eval loop. Attach it
    afterwards from the training sequences if you need cold-start bucketing.
    """
    model.eval()
    if ks is not None:
        ks = tuple(sorted(ks))
    else:
        ks = (int(k),)
    k_max = max(ks)

    cum_recall = {kk: 0.0 for kk in ks}
    cum_ndcg = {kk: 0.0 for kk in ks}
    total_samples = 0

    all_scores = []
    all_labels = []
    all_hist_items = []
    all_hist_behs = []

    with torch.no_grad():
        for batch in eval_loader:
            batch = {kk: v.to(device) for kk, v in batch.items()}
            query = model(batch, need_pooling=True)
            if query.dim() != 2:
                raise ValueError(
                    f"Expected query to be 2D [batch_size, embed_dim], got {query.shape}"
                )

            scores = query @ model.item_embedding.weight.T
            in_items = batch['in_item_id']
            scores.scatter_(1, in_items, -1e9)

            targets = batch['item_id']
            valid = targets > 0
            if not valid.any():
                continue
            scores_v = scores[valid]
            targets_v = targets[valid]

            _, topk_idx = torch.topk(scores_v, k=k_max, dim=1)
            hits = topk_idx == targets_v.unsqueeze(1)
            hit_any = hits.any(dim=1)
            ranks = torch.where(
                hit_any,
                hits.float().argmax(dim=1),
                torch.full_like(hits.float().argmax(dim=1), k_max),
            )

            n_valid = int(targets_v.size(0))
            total_samples += n_valid
            log2_discount = 1.0 / torch.log2(ranks.float() + 2.0)

            for kk in ks:
                hit_at_k = hit_any & (ranks < kk)
                cum_recall[kk] += float(hit_at_k.float().sum().item())
                ndcg_vals = torch.where(
                    hit_at_k, log2_discount, torch.zeros_like(log2_discount)
                )
                cum_ndcg[kk] += float(ndcg_vals.sum().item())

            # collect for bucketed analysis
            all_scores.append(topk_idx.cpu().numpy().astype(np.int64))
            all_labels.append(targets_v.cpu().numpy().astype(np.int64))
            all_hist_items.append(in_items.cpu().numpy().astype(np.int64))
            if 'in_behavior_id' in batch:
                all_hist_behs.append(
                    batch['in_behavior_id'].cpu().numpy().astype(np.int64)
                )
            else:
                all_hist_behs.append(
                    np.zeros_like(in_items.cpu().numpy(), dtype=np.int64)
                )

    dump_path = Path(dump_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dump_path,
        scores=np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0, k_max), dtype=np.int64),
        labels=np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64),
        hist_items=np.concatenate(all_hist_items, axis=0) if all_hist_items else np.zeros((0, 0), dtype=np.int64),
        hist_behs=np.concatenate(all_hist_behs, axis=0) if all_hist_behs else np.zeros((0, 0), dtype=np.int64),
        user_idx=np.arange(sum(s.shape[0] for s in all_scores), dtype=np.int64),
    )

    if total_samples == 0:
        out = {}
        for kk in ks:
            out[f'recall@{kk}'] = 0.0
            out[f'ndcg@{kk}'] = 0.0
        return out

    out = {}
    for kk in ks:
        out[f'recall@{kk}'] = cum_recall[kk] / total_samples
        out[f'ndcg@{kk}'] = cum_ndcg[kk] / total_samples
    return out


def mrr_at_k(preds, targets, k=10):
    """计算MRR@k"""
    # 取top-k预测
    _, topk_indices = torch.topk(preds, k, dim=-1)  # [batch_size, k]
    
    if targets.dim() == 1:
        # 单目标情况
        expanded_targets = targets.unsqueeze(-1)  # [batch_size, 1]
        hits = (topk_indices == expanded_targets)  # [batch_size, k]
    else:
        # 多目标情况
        batch_indices = torch.arange(preds.size(0), device=preds.device).unsqueeze(1).expand(-1, k)
        target_mask = targets > 0
        hits = target_mask[batch_indices, topk_indices]  # [batch_size, k]
    
    # 计算第一次命中的位置
    first_hit_pos = torch.where(hits, 
                               torch.arange(1, k+1, device=preds.device).float().unsqueeze(0),
                               torch.zeros(1, device=preds.device)).max(dim=1)[0]  # [batch_size]
    
    # 计算倒数
    mrr = torch.zeros_like(first_hit_pos)
    hit_mask = first_hit_pos > 0
    mrr[hit_mask] = 1.0 / first_hit_pos[hit_mask]
    
    return mrr.mean()

def dr4sr_evaluate(model, eval_loader, top_k=10, device=None, item_embeddings=None):
    """DR4SR风格的评估函数"""
    model.eval()
    
    cum_recall = 0.0
    cum_ndcg = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 获取查询表示
            query = model(batch)  # [batch_size, embed_dim]
            
            # 计算与所有物品的得分
            scores = query @ model.item_embedding.weight.T  # [batch_size, num_items]
            
            # 屏蔽历史物品
            for i in range(batch['in_item_id'].size(0)):
                hist_items = batch['in_item_id'][i][batch['in_item_id'][i] > 0]
                scores[i, hist_items] = -1e9
            
            # 获取目标物品
            targets = batch['item_id']  # [batch_size]
            
            # 计算TopK
            _, topk_indices = torch.topk(scores, k=top_k, dim=1)  # [batch, k]
            
            # 计算指标
            for i in range(targets.size(0)):
                target = targets[i].item()
                if target == 0:  # 跳过padding
                    continue
                
                # Recall@K
                hit = (topk_indices[i] == target).any().item()
                recall = 1.0 if hit else 0.0
                cum_recall += recall
                
                # NDCG@K
                if hit:
                    rank = torch.where(topk_indices[i] == target)[0].item()
                    ndcg = 1.0 / np.log2(rank + 2)
                else:
                    ndcg = 0.0
                cum_ndcg += ndcg
                
                total_samples += 1
    
    if total_samples == 0:
        return {'recall@10': 0.0, 'ndcg@10': 0.0}
    
    return {
        f'recall@{top_k}': cum_recall / total_samples,
        f'ndcg@{top_k}': cum_ndcg / total_samples
    }