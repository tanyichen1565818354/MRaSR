import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_initialization(module, initial_range=0.02):
    """N(0, initial_range) 权重初始化"""
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class BinaryCrossEntropyLoss(nn.Module):
    """加权二元交叉熵（正样本 + 负样本）"""
    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score, reduce=True):
        # pos_score: B | B x L | B x L
        # neg_score: B x neg | B x L x neg | B x neg
        
        weight = self._cal_weight(neg_score)
        padding_mask = torch.isinf(pos_score)
        # positive
        pos_loss = F.logsigmoid(pos_score)
        pos_loss.masked_fill_(padding_mask, 0.0)
        if reduce:
            pos_loss = pos_loss.sum() / (~padding_mask).sum()
        else:
            pos_loss = pos_loss / (~padding_mask).sum()
        # negative
        neg_loss = F.softplus(neg_score) * weight
        neg_loss = neg_loss.sum(-1)
        # mask
        if pos_score.dim() == neg_score.dim()-1:
            neg_loss.masked_fill_(padding_mask, 0.0)
            if reduce:
                neg_loss = neg_loss.sum() / (~padding_mask).sum()
            else:
                neg_loss = neg_loss / (~padding_mask).sum()
        else:
            neg_loss = torch.mean(neg_loss)
        return -pos_loss + neg_loss

    def _cal_weight(self, neg_score):
        return torch.ones_like(neg_score) / neg_score.size(-1)

class BaseModel(nn.Module):
    """序列推荐基类"""
    
    def __init__(self, config, dataset_list=None):
        super().__init__()
        # 配置
        self.config = config
        self.dataset_list = dataset_list or []
        
        # 基础属性
        self.fuid = 'user_id'
        self.fiid = 'item_id'
        
        # 从配置中获取参数
        self.embed_dim = config.get('model', {}).get('embed_dim', 64)
        self.max_seq_len = config.get('data', {}).get('max_seq_len', 50)
        
        # 物品数量 - 支持多种配置方式
        if dataset_list and len(dataset_list) > 0:
            self.num_items = dataset_list[0].num_items if hasattr(dataset_list[0], 'num_items') else config.get('num_items', 1000)
        else:
            self.num_items = config.get('num_items', 1000)
        
        # 设备配置
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 物品嵌入
        self.item_embedding = nn.Embedding(self.num_items, self.embed_dim, padding_idx=0)
        
        # 损失函数
        self.loss_fn = self._get_loss_func()
        
    def _get_loss_func(self):
        """获取损失函数"""
        loss_type = self.config.get('model', {}).get('loss_fn', 'bce')
        if loss_type == 'bce':
            return BinaryCrossEntropyLoss()
        else:
            return BinaryCrossEntropyLoss()  # 默认使用BCE
    
    def _neg_sampling(self, batch):
        """负采样：均匀采样避开 padding=0。

        原实现每 batch 都分配 ``[B, num_items]`` 的全 1 权重矩阵再走 multinomial，
        在大词表（QK-video ~10w 物品）下每 batch 要 100MB+ 且反复分配。
        由于权重恒为 1（除 padding 列），等价于在 ``[1, num_items)`` 上 randint。
        """
        target = batch[self.fiid]
        n_neg = self.max_seq_len if target.dim() == 2 else 1
        neg_idx = torch.randint(
            1, self.num_items, (target.shape[0], n_neg), device=self.device
        )
        return neg_idx.reshape_as(target).unsqueeze(-1)

    def forward(self, batch, need_pooling=True):
        """前向传播 - 子类重写"""
        raise NotImplementedError

    def training_step(self, batch, reduce=True, return_query=False):
        """训练一步：正负样本打分并计算 BCE"""
        query = self.forward(batch)
        
        # 处理不同维度的query（SASRec训练时使用origin池化返回3D）
        if query.dim() == 3:
            # 如果query是3D [batch_size, seq_len, embed_dim]，取最后一个位置
            seq_len = batch['seqlen']
            gather_index = (seq_len - 1).view(-1, 1, 1).expand(-1, -1, query.size(-1))
            query = query.gather(dim=1, index=gather_index).squeeze(1)  # [batch_size, embed_dim]
        
        # 正负样本内积打分
        pos_score = (query * self.item_embedding.weight[batch[self.fiid]]).sum(-1)
        neg_score = (query.unsqueeze(-2) * self.item_embedding.weight[batch['neg_item']]).sum(-1)
        pos_score[batch[self.fiid] == 0] = -torch.inf  # padding

        loss_value = self.loss_fn(pos_score, neg_score, reduce=reduce)
        if return_query:
            return loss_value, query
        else:
            return loss_value
        
    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state': self.state_dict(),
            'config': self.config
        }, path)
        
    @classmethod
    def load(cls, path):
        """加载模型"""
        ckpt = torch.load(path, weights_only=False)
        model = cls(ckpt['config'])
        model.load_state_dict(ckpt['model_state'])
        return model