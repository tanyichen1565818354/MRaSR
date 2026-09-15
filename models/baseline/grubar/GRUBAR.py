"""GRUBAR: GRU4Rec 的多行为扩展。

架构：
  输入 = item_embedding + behavior_embedding
  -> 多层 GRU
  -> [B, L, H]
  -> Linear(H, D)
  -> [B, L, D]

训练时返回 [B, L, D]（origin 池化），training_step 取最后位置 + 目标行为条件化。
评估时返回 [B, D]（最后位置 + 目标行为 like 条件化），直接用 self.item_embedding 打分。
"""
import torch
import torch.nn as nn

from models.baseline.mb_base import MultiBehaviorBaseModel


class GRUBAR(MultiBehaviorBaseModel):
    def __init__(self, config, dataset_list=None):
        super().__init__(config, dataset_list)
        m = config['model']

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

        emb = self.item_embedding(user_hist) + self.behavior_embedding(beh)  # [B, L, D]
        gru_out, _ = self.gru(emb)            # [B, L, H]
        out = self.output_proj(gru_out)       # [B, L, D]
        return out
