import torch
import torch.nn as nn

class SeqPoolingLayer(nn.Module):
    """支持多种池化策略的序列池化层"""
    def __init__(self, pooling_type='mean', keepdim=False, config=None):
        super().__init__()
        valid_types = ['origin', 'mask', 'concat', 'sum', 'mean', 'max', 'last']
        if pooling_type not in valid_types:
            raise ValueError(f"Invalid pooling_type: {pooling_type}, must be one of {valid_types}")
        
        # 可以从config中读取默认参数
        if config and hasattr(config, 'pooling'):
            self.pooling_type = config.pooling.get('type', pooling_type)
            self.keepdim = config.pooling.get('keepdim', keepdim)
        else:
            self.pooling_type = pooling_type
            self.keepdim = keepdim

    def forward(self, batch_seq_embeddings, seq_len, weight=None, mask_token=None):
        """
        输入：
            batch_seq_embeddings: [B, L, D] 批次序列嵌入
            seq_len: [B] 每个序列的实际长度
            weight: [B, L] 可选权重（用于加权池化）
            mask_token: 用于mask池化的特殊标记
        输出：
            pooled: 池化后的结果，形状取决于池化类型
        """
        B, L, D = batch_seq_embeddings.size()
        device = batch_seq_embeddings.device

        # 获取有效范围掩码
        range_tensor = torch.arange(L).unsqueeze(0).to(device)  # [1, L]
        valid_mask = (range_tensor < seq_len.unsqueeze(1))      # [B, L]

        if self.pooling_type == 'origin':
            # 原始序列返回（可选保持维度）
            return batch_seq_embeddings if self.keepdim else batch_seq_embeddings.reshape(B, -1)

        elif self.pooling_type == 'mask':
            # 用mask_token填充无效位置后返回
            if mask_token is None:
                mask_token = torch.zeros(D).to(device)
            masked_emb = torch.where(valid_mask.unsqueeze(-1), 
                                   batch_seq_embeddings,
                                   mask_token)
            return masked_emb

        elif self.pooling_type == 'concat':
            # 拼接池化结果和最后元素
            pooled = self._base_pooling(batch_seq_embeddings, valid_mask, 'mean')
            last_items = batch_seq_embeddings[torch.arange(B), seq_len-1]
            return torch.cat([pooled, last_items], dim=-1)

        elif self.pooling_type in ['sum', 'mean', 'max']:
            return self._base_pooling(batch_seq_embeddings, valid_mask, self.pooling_type)

        elif self.pooling_type == 'last':
            # 提取每个序列最后一个有效项
            last_indices = seq_len - 1
            pooled = batch_seq_embeddings[torch.arange(B), last_indices]
            return pooled.unsqueeze(1) if self.keepdim else pooled

    def _base_pooling(self, embeddings, mask, mode):
        """基础池化操作"""
        masked_emb = embeddings * mask.unsqueeze(-1)
        
        if mode == 'sum':
            pooled = masked_emb.sum(dim=1)
        elif mode == 'mean':
            sum_emb = masked_emb.sum(dim=1)
            pooled = sum_emb / mask.sum(dim=1, keepdim=True)
        elif mode == 'max':
            pooled, _ = masked_emb.max(dim=1)
        
        return pooled.unsqueeze(1) if self.keepdim else pooled
