"""RLBL: Recurrent Log-BiLinear model（多行为序列推荐的开创性工作）。

核心思想：
  将 RNN 与 log-bilinear（LBL）结合以同时建模长期与短期偏好，并使用
  behavior-specific transition matrices 区分异构行为。具体地：
    1. 把序列按窗口大小 W 切分，每个窗口内用 LBL 捕获短期上下文：
         h_t = sum_{i=1}^{W} C_{beh_{t-i}}[i-1] @ item_emb_{t-i}
       其中转移矩阵 C 按源位置的「行为 id」索引（behavior-specific）。
    2. 把每个位置得到的短期表示送入 GRU，聚合为长期上下文。
    3. Linear 投影回 embed_dim。

训练/评估均直接返回最后有效位置的 [B, D]；评估时再加目标行为条件化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from models.baseline.mb_base import MultiBehaviorBaseModel


class RLBL(MultiBehaviorBaseModel):
    def __init__(self, config, dataset_list=None):
        super().__init__(config, dataset_list)
        m = config['model']

        # 窗口大小 W：LBL 的上下文长度
        self.window_size = int(m.get('window_size', 2))
        # 行为特定 + 位置特定的转移矩阵：C[beh, pos, D, D]
        # 索引 0 行为 = padding，对应转移置零
        self.transition = nn.Parameter(
            torch.empty(self.num_behaviors + 1, self.window_size, self.embed_dim, self.embed_dim)
        )
        nn.init.xavier_uniform_(self.transition[1:])
        nn.init.zeros_(self.transition[0])

        # 短期表示维度 = embed_dim，GRU 聚合为长期上下文
        self.hidden_size = int(m.get('hidden_size', 128))
        self.layer_num = int(m.get('layer_num', 2))
        dropout = float(m.get('dropout_rate', 0.5))

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            batch_first=True,
            dropout=dropout if self.layer_num > 1 else 0,
        )
        self.output_proj = nn.Linear(self.hidden_size, self.embed_dim)

        self.apply(self._init_weights)
        # apply 会覆盖手工初始化的转移矩阵，重新设置
        nn.init.xavier_uniform_(self.transition[1:])
        nn.init.zeros_(self.transition[0])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight.data[module.padding_idx], 0.0)

    def _apply_transition(self, emb: torch.Tensor, beh: torch.Tensor, pos: int) -> torch.Tensor:
        """按行为应用转移矩阵，避免物化 [B, L, D, D]。

        原实现 ``trans[beh]`` 会分配巨大的中间张量；这里对每种行为做一次
        ``[B,L,D] @ [D,D]`` GEMM，再用 mask 累加（行为种类通常 ≤ 8）。
        """
        # emb: [B, L, D], beh: [B, L], transition[:, pos]: [nb+1, D, D]
        out = torch.zeros_like(emb)
        trans_pos = self.transition[:, pos]  # [nb+1, D, D]
        # 跳过 padding 行为 0（转移已置零）
        for b in range(1, self.num_behaviors + 1):
            mask = (beh == b).unsqueeze(-1)  # [B, L, 1]
            # C @ e  ≡  e @ C^T；不做 .any() 以免每步 GPU sync
            out = out + torch.matmul(emb, trans_pos[b].transpose(0, 1)) * mask
        return out

    def _encode(self, batch) -> torch.Tensor:
        user_hist = batch['in_item_id']       # [B, L]
        beh = batch.get('in_behavior_id')     # [B, L], 已 +1 偏移
        if beh is None:
            beh = torch.ones_like(user_hist)

        B, L = user_hist.shape
        W = self.window_size
        item_emb = self.item_embedding(user_hist)  # [B, L, D]

        # ---- LBL 短期上下文（平移已有 embedding，避免重复查表）----
        short_repr = torch.zeros_like(item_emb)
        for i in range(1, W + 1):
            # 源位置 t-i：右移 i 位，左侧补零
            src_emb = F.pad(item_emb[:, :-i], (0, 0, i, 0))   # [B, L, D]
            src_beh = F.pad(beh[:, :-i], (i, 0), value=0)     # [B, L]
            short_repr = short_repr + self._apply_transition(src_emb, src_beh, i - 1)

        # ---- GRU 聚合：pack 跳过 padding，直接取最后隐状态 ----
        lengths = batch['seqlen'].detach().clamp(min=1).cpu()
        packed = pack_padded_sequence(
            short_repr, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)          # h_n: [num_layers, B, H]
        return self.output_proj(h_n[-1])   # [B, D]

    def forward(self, batch, need_pooling=True):
        query = self._encode(batch)  # [B, D]
        if not self.training:
            target_beh = batch.get('behavior_id')
            if target_beh is not None:
                query = query + self.behavior_embedding(target_beh)
        return query
