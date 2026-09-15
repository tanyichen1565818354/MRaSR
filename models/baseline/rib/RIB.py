"""RIB: Recommendation from the micro-behavior perspective（可解释多行为框架）。

核心思想：
  1. 输入为 (item, behavior) 对序列，分别通过 embedding 层得到 item 嵌入与
     behavior 嵌入，拼接为 [B, L, 2D]。
  2. 送入 GRU 得到每个时间步的隐状态 [B, L, H]。
  3. 引入行为感知的注意力机制：用 behavior 嵌入作为 query 对 GRU 隐状态做
     加权聚合，使不同行为对最终表示的贡献可解释。
  4. 残差融合 item 嵌入后投影回 embed_dim。

训练/评估均直接返回最后有效位置的 [B, D]；评估时再加目标行为条件化。
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.baseline.mb_base import MultiBehaviorBaseModel


class RIB(MultiBehaviorBaseModel):
    def __init__(self, config, dataset_list=None):
        super().__init__(config, dataset_list)
        m = config['model']

        self.hidden_size = int(m.get('hidden_size', 128))
        self.layer_num = int(m.get('layer_num', 2))
        dropout = float(m.get('dropout_rate', 0.5))

        # 输入 = concat(item_emb, behavior_emb) -> 2 * embed_dim
        self.gru = nn.GRU(
            input_size=2 * self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            batch_first=True,
            dropout=dropout if self.layer_num > 1 else 0,
        )
        # 行为感知注意力：用 behavior 嵌入作为 query 打分
        self.attn_proj = nn.Linear(self.embed_dim, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.embed_dim)
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
        item_emb = self.item_embedding(user_hist)        # [B, L, D]
        beh_emb = self.behavior_embedding(beh)           # [B, L, D]
        x = torch.cat([item_emb, beh_emb], dim=-1)      # [B, L, 2D]
        x = self.dropout(x)

        # pack 跳过 padding，加速变长序列
        lengths = batch['seqlen'].detach().clamp(min=1)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(
            gru_out, batch_first=True, total_length=L
        )  # [B, L, H]

        # ---- 行为感知注意力 ----
        mask = user_hist == 0  # [B, L] True = pad
        query = self.attn_proj(beh_emb)                  # [B, L, H]
        scores = (query * gru_out).sum(-1) / (self.hidden_size ** 0.5)  # [B, L]
        scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)             # [B, L]
        attn = torch.nan_to_num(attn, nan=0.0)           # 全 pad 序列兜底
        context = torch.bmm(attn.unsqueeze(1), gru_out).squeeze(1)  # [B, H]

        # 只取最后有效位置，避免对整段序列做投影
        last_idx = (lengths - 1).to(device=gru_out.device)
        batch_idx = torch.arange(B, device=gru_out.device)
        last_h = gru_out[batch_idx, last_idx]            # [B, H]
        last_item = item_emb[batch_idx, last_idx]        # [B, D]
        out = self.output_proj(last_h + context) + last_item  # [B, D]
        return out

    def forward(self, batch, need_pooling=True):
        query = self._encode(batch)  # [B, D]
        if not self.training:
            target_beh = batch.get('behavior_id')
            if target_beh is not None:
                query = query + self.behavior_embedding(target_beh)
        return query
