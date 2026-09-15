"""MBHT: Multi-Behavior Heterogeneous Transformer。

与 SASBAR/GRUBAR 的关键区别：
  1. 行为特定物品表示（heterogeneous item representations）：
     每个行为 b 有独立的线性变换 W_b，物品在不同行为下产生不同的表示。
  2. 行为类型化注意力偏置（behavior-typed attention bias）：
     位置 i 到位置 j 的注意力额外加上一个可学习的行为转移矩阵 R[beh_i, beh_j]，
     捕获"行为 j 的物品对行为 i 的查询有多重要"。
  3. 自定义多头注意力以支持 per-batch 的 [B, L, L] 行为偏置。

共享 self.item_embedding 用于评分与导出（融合表示）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline.Baseline import normal_initialization
from models.baseline.mb_base import MultiBehaviorBaseModel


class BehaviorAwareAttention(nn.Module):
    """支持 per-batch 加性偏置 [B, L, L] 的多头注意力。"""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0, f"d_model={d_model} 必须能被 n_head={n_head} 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x, causal_mask, padding_mask, behavior_bias):
        # x: [B, L, D]
        # causal_mask: [L, L] bool, True = 屏蔽
        # padding_mask: [B, L] bool, True = pad
        # behavior_bias: [B, L, L] 加性 float
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each [B, L, n_head, d_head]
        q = q.transpose(1, 2)  # [B, n_head, L, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, n_head, L, L]
        if behavior_bias is not None:
            attn = attn + behavior_bias.unsqueeze(1)  # broadcast 到 n_head

        causal = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)
        attn = attn.masked_fill(causal, float('-inf'))

        if padding_mask is not None:
            pad = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L] 屏蔽 key 的 pad
            attn = attn.masked_fill(pad, float('-inf'))

        prob = torch.softmax(attn, dim=-1)
        prob = self.dropout(prob)
        out = prob @ v  # [B, n_head, L, d_head]
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class MBHTBlock(nn.Module):
    def __init__(self, d_model, n_head, hidden_size, dropout, layer_norm_eps):
        super().__init__()
        self.attn = BehaviorAwareAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, causal_mask, padding_mask, behavior_bias):
        x = x + self.attn(self.norm1(x), causal_mask, padding_mask, behavior_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class MBHT(MultiBehaviorBaseModel):
    def __init__(self, config, dataset_list=None):
        super().__init__(config, dataset_list)
        m = config['model']

        # 行为特定物品变换：W[b] @ item_emb + b[b]
        # 形状 [num_behaviors+1, D_out, D_in] 与 [num_behaviors+1, D]
        self.behavior_transform_weight = nn.Parameter(
            torch.empty(self.num_behaviors + 1, self.embed_dim, self.embed_dim)
        )
        self.behavior_transform_bias = nn.Parameter(
            torch.zeros(self.num_behaviors + 1, self.embed_dim)
        )
        nn.init.xavier_uniform_(self.behavior_transform_weight[1:])
        nn.init.zeros_(self.behavior_transform_bias[1:])
        # padding 行为 (index 0) 的变换置零
        nn.init.zeros_(self.behavior_transform_weight[0])
        nn.init.zeros_(self.behavior_transform_bias[0])

        # 行为转移矩阵 R[query_beh, key_beh]，作为注意力加性偏置
        self.behavior_relation = nn.Parameter(
            torch.zeros(self.num_behaviors + 1, self.num_behaviors + 1)
        )

        self.position_emb = nn.Embedding(self.max_seq_len, self.embed_dim)
        self.dropout = nn.Dropout(float(m.get('dropout_rate', 0.5)))

        self.blocks = nn.ModuleList([
            MBHTBlock(
                self.embed_dim,
                int(m.get('head_num', 2)),
                int(m.get('hidden_size', 128)),
                float(m.get('dropout_rate', 0.5)),
                float(m.get('layer_norm_eps', 1e-12)),
            )
            for _ in range(int(m.get('layer_num', 2)))
        ])

        self.apply(normal_initialization)
        # apply 会覆盖我们手工初始化的行为变换/转移参数，重新设置
        nn.init.xavier_uniform_(self.behavior_transform_weight[1:])
        nn.init.zeros_(self.behavior_transform_weight[0])
        nn.init.zeros_(self.behavior_transform_bias)
        nn.init.zeros_(self.behavior_relation)

    def _encode(self, batch) -> torch.Tensor:
        user_hist = batch['in_item_id']       # [B, L]
        beh = batch.get('in_behavior_id')     # [B, L], 已 +1 偏移
        if beh is None:
            beh = torch.ones_like(user_hist)

        B, L = user_hist.shape
        item_emb = self.item_embedding(user_hist)  # [B, L, D]

        # 行为特定物品变换：het_rep[b,l,o] = sum_i W[beh[b,l], o, i] * item_emb[b,l,i] + bias[beh[b,l], o]
        # 旧实现 ``self.behavior_transform_weight[beh]`` 会 gather 出 [B, L, D, D] 中间张量
        # （B*L*D*D = 52M 元素 / 209MB），是 MBHT 的主要内存与显存瓶颈。
        # 这里用 one-hot 把"按行为选 W"融进 einsum，中间张量降为 [B, L, nb+1]（100K 元素）。
        onehot = F.one_hot(beh, self.num_behaviors + 1).float()  # [B, L, nb+1]
        het_rep = torch.einsum(
            'noi,bli,bln->blo',
            self.behavior_transform_weight,  # [nb+1, D, D]
            item_emb,                        # [B, L, D]
            onehot,                          # [B, L, nb+1]
        )
        het_rep = het_rep + self.behavior_transform_bias[beh]  # [B, L, D]

        positions = torch.arange(L, dtype=torch.long, device=user_hist.device).unsqueeze(0).expand_as(user_hist)
        x = het_rep + self.behavior_embedding(beh) + self.position_emb(positions)
        x = self.dropout(x)

        # 行为偏置 [B, L, L] = R[beh_query, beh_key]
        # 旧实现用 beh_q.expand(B,L,L), beh_k.expand(B,L,L) 做双 advanced-indexing，
        # 会 materialize 两个 [B,L,L] int64 拷贝。改用两次 gather 避免大拷贝。
        row = self.behavior_relation[beh]  # [B, L, nb+1]
        behavior_bias = row.gather(
            2, beh.unsqueeze(1).expand(B, L, L)
        )  # [B, L, L]

        causal_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        padding_mask = user_hist == 0  # [B, L]

        for block in self.blocks:
            x = block(x, causal_mask, padding_mask, behavior_bias)
        return x  # [B, L, D]
