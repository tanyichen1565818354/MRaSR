"""SASBAR: SASRec 的多行为扩展。

架构：
  输入 = item_embedding + behavior_embedding + position_embedding
  -> 因果掩码 Transformer Encoder
  -> [B, L, D]

训练时返回 [B, L, D]（origin 池化），training_step 取最后位置 + 目标行为条件化。
评估时返回 [B, D]（最后位置 + 目标行为 like 条件化），直接用 self.item_embedding 打分。
"""
import torch
import torch.nn as nn

from models.baseline.Baseline import normal_initialization
from models.baseline.mb_base import MultiBehaviorBaseModel


class SASBAR(MultiBehaviorBaseModel):
    def __init__(self, config, dataset_list=None):
        super().__init__(config, dataset_list)
        m = config['model']

        self.position_emb = nn.Embedding(self.max_seq_len, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(m.get('head_num', 2)),
            dim_feedforward=int(m.get('hidden_size', 128)),
            dropout=float(m.get('dropout_rate', 0.5)),
            activation=m.get('activation', 'gelu'),
            layer_norm_eps=float(m.get('layer_norm_eps', 1e-12)),
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(m.get('layer_num', 2)))
        self.dropout = nn.Dropout(float(m.get('dropout_rate', 0.5)))

        self.apply(normal_initialization)

    def _encode(self, batch) -> torch.Tensor:
        user_hist = batch['in_item_id']            # [B, L]
        beh = batch.get('in_behavior_id')          # [B, L], 已 +1 偏移
        if beh is None:
            beh = torch.ones_like(user_hist)

        L = user_hist.size(1)
        positions = torch.arange(L, dtype=torch.long, device=user_hist.device).unsqueeze(0).expand_as(user_hist)

        seq_embs = self.item_embedding(user_hist) + self.behavior_embedding(beh) + self.position_emb(positions)
        mask4padding = user_hist == 0  # [B, L]
        attn_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)  # 因果

        out = self.transformer(src=self.dropout(seq_embs), mask=attn_mask, src_key_padding_mask=mask4padding)
        return out  # [B, L, D]
