"""BINN: Behavior-Intensive Neural Network（KDD 2018）。

核心思想：
  区分性建模用户「短期会话动机」与「长期稳定偏好」：
    1. CLSTM（行为条件化 LSTM）：把 behavior 嵌入与 item 嵌入拼接后送入 LSTM，
       学习会话内的短期消费动机（SBL, Session Behavior Learning）。
    2. Bi-CLSTM（双向行为条件化 LSTM）：仅保留「目标行为」位置的物品嵌入，
       其余位置置零，双向 LSTM 学习长期稳定偏好（PBL, Preference Behavior Learning）。
    3. 拼接 SBL 与 PBL 的隐状态，投影回 embed_dim。

训练/评估均直接返回最后有效位置的 [B, D]；评估时再加目标行为条件化。
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.baseline.mb_base import MultiBehaviorBaseModel


class BINN(MultiBehaviorBaseModel):
    def __init__(self, config, dataset_list=None):
        super().__init__(config, dataset_list)
        m = config['model']

        self.hidden_size = int(m.get('hidden_size', 128))
        self.layer_num = int(m.get('layer_num', 2))
        dropout = float(m.get('dropout_rate', 0.5))

        # CLSTM（SBL）：输入 = concat(item_emb, behavior_emb) -> 2D
        self.clstm = nn.LSTM(
            input_size=2 * self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            batch_first=True,
            dropout=dropout if self.layer_num > 1 else 0,
        )
        # Bi-CLSTM（PBL）：仅目标行为位置保留 item 嵌入，其余置零
        self.bi_clstm = nn.LSTM(
            input_size=2 * self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if self.layer_num > 1 else 0,
        )
        # 融合 SBL(H) + PBL(2H) -> D
        self.output_proj = nn.Linear(self.hidden_size + 2 * self.hidden_size, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight.data[module.padding_idx], 0.0)

    def _encode(self, batch) -> torch.Tensor:
        user_hist = batch['in_item_id']       # [B, L]
        beh = batch.get('in_behavior_id')     # [B, L]
        if beh is None:
            beh = torch.ones_like(user_hist)

        B, L = user_hist.shape
        item_emb = self.item_embedding(user_hist)   # [B, L, D]
        beh_emb = self.behavior_embedding(beh)     # [B, L, D]
        lengths = batch['seqlen'].detach().clamp(min=1)
        lengths_cpu = lengths.cpu()

        # ---- SBL：CLSTM；单向可直接取最后隐状态，无需 pad 回全序列 ----
        sbl_input = self.dropout(torch.cat([item_emb, beh_emb], dim=-1))
        sbl_packed = pack_padded_sequence(
            sbl_input, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, (sbl_h, _) = self.clstm(sbl_packed)   # sbl_h: [num_layers, B, H]
        sbl_last = sbl_h[-1]                     # [B, H]

        # ---- PBL：Bi-CLSTM；需逐步输出以取最后有效位置的双向表示 ----
        target_beh_id = self.target_behavior + 1
        pref_mask = (beh == target_beh_id).unsqueeze(-1).to(dtype=item_emb.dtype)
        pref_item_emb = item_emb * pref_mask
        pbl_input = self.dropout(torch.cat([pref_item_emb, beh_emb], dim=-1))
        pbl_packed = pack_padded_sequence(
            pbl_input, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        pbl_out, _ = self.bi_clstm(pbl_packed)
        pbl_out, _ = pad_packed_sequence(
            pbl_out, batch_first=True, total_length=L
        )  # [B, L, 2H]

        last_idx = (lengths - 1).to(device=pbl_out.device)
        batch_idx = torch.arange(B, device=pbl_out.device)
        pbl_last = pbl_out[batch_idx, last_idx]  # [B, 2H]

        fused = torch.cat([sbl_last, pbl_last], dim=-1)  # [B, 3H]
        return self.output_proj(fused)                   # [B, D]

    def forward(self, batch, need_pooling=True):
        query = self._encode(batch)  # [B, D]
        if not self.training:
            target_beh = batch.get('behavior_id')
            if target_beh is not None:
                query = query + self.behavior_embedding(target_beh)
        return query
