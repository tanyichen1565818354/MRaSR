"""多行为序列推荐 baseline 的公共基类。

三个多行为 baseline（SASBAR / GRUBAR / MBHT）共享以下逻辑：
  - 行为嵌入（behavior_embedding，padding_idx=0，行为 id 已 +1 偏移）
  - 训练时 forward 返回 [B, L, D]（origin 池化），由 training_step 取最后位置
    并用「目标行为」embedding 条件化查询向量
  - 评估时 forward 返回 [B, D]（最后位置 + 目标行为条件化），供 eval 直接打分
  - self.item_embedding 为共享物品嵌入（评分与导出共用）
  - get_item_embedding() 导出融合后的 item embedding 供再生器使用
子类只需实现 _encode(batch) -> [B, L, D]。
"""
import torch
import torch.nn as nn

from models.baseline.Baseline import BaseModel, normal_initialization


class MultiBehaviorBaseModel(BaseModel):
    """多行为 baseline 公共基类。"""

    def __init__(self, config, dataset_list=None):
        super().__init__(config, dataset_list)
        model_cfg = config.get('model', {})

        # 行为相关参数
        # num_behaviors 为「行为种类数」（不含 padding），嵌入表大小 = num_behaviors + 1
        self.num_behaviors = int(model_cfg.get('num_behaviors', 7))
        # 目标行为 id（原始编码，如 like=1）；运行时行为 id 已 +1 偏移
        self.target_behavior = int(model_cfg.get('target_behavior', 1))

        # 行为嵌入：索引 0 = padding，1..num_behaviors = 各行为
        self.behavior_embedding = nn.Embedding(self.num_behaviors + 1, self.embed_dim, padding_idx=0)

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------
    def _encode(self, batch) -> torch.Tensor:
        """返回上下文化后的序列表示 [B, L, D]。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(self, batch, need_pooling=True):
        seq_repr = self._encode(batch)  # [B, L, D]

        if self.training:
            # 训练：返回 [B, L, D]，由 training_step 取最后位置 + 条件化
            return seq_repr

        # 评估/推理：取最后非 pad 位置 + 目标行为条件化 -> [B, D]
        seqlen = batch['seqlen']
        gather_index = (seqlen - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, seq_repr.size(-1))
        query = seq_repr.gather(dim=1, index=gather_index).squeeze(1)  # [B, D]
        target_beh = batch.get('behavior_id')
        if target_beh is not None:
            query = query + self.behavior_embedding(target_beh)
        return query

    # ------------------------------------------------------------------
    # 训练步骤（覆盖 BaseModel，加入目标行为条件化）
    # ------------------------------------------------------------------
    def training_step(self, batch, reduce=True, return_query=False):
        query = self.forward(batch)  # [B, L, D] in train

        if query.dim() == 3:
            seq_len = batch['seqlen']
            gather_index = (seq_len - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, query.size(-1))
            query = query.gather(dim=1, index=gather_index).squeeze(1)  # [B, D]

        # 用「下一个 item 的行为」条件化查询：预测给定历史与目标行为下的下一个物品
        target_beh = batch.get('behavior_id')
        if target_beh is not None:
            query = query + self.behavior_embedding(target_beh)

        pos_score = (query * self.item_embedding.weight[batch[self.fiid]]).sum(-1)
        neg_score = (query.unsqueeze(-2) * self.item_embedding.weight[batch['neg_item']]).sum(-1)
        pos_score[batch[self.fiid] == 0] = -torch.inf  # padding

        loss_value = self.loss_fn(pos_score, neg_score, reduce=reduce)
        if return_query:
            return loss_value, query
        return loss_value

    # ------------------------------------------------------------------
    # 导出 item embedding（融合后的共享表示）
    # ------------------------------------------------------------------
    def get_item_embedding(self):
        return self.item_embedding.weight.detach().cpu()

    def save(self, path):
        torch.save({'model_state': self.state_dict(), 'config': self.config}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(ckpt['config'])
        model.load_state_dict(ckpt['model_state'])
        return model
