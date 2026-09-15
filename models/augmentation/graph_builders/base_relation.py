from abc import ABC, abstractmethod
from collections import defaultdict
import logging
import torch

class BaseRelationBuilder(ABC):
    def __init__(self, products, sequences, item2idx, config=None): 
        self.products = products
        self.sequences = sequences
        self.item2idx = item2idx
        self.edge_index = defaultdict(float)
        self.config = config
        self.idx2item = {v: k for k, v in item2idx.items()}
        self.logger = logging.getLogger(__name__)
        
        # 🔧 修正：获取特殊token索引
        self.special_token_indices = self._get_special_token_indices()
        self.logger.info(f"特殊token索引: {self.special_token_indices}")
        
    def _get_special_token_indices(self):
        """获取特殊token的索引"""
        special_indices = set()
        for asin, idx in self.item2idx.items():
            if asin == '<PAD>' or idx == 0:
                special_indices.add(idx)
        return special_indices
        
    @abstractmethod
    def build(self):
        """必须实现的构建方法"""
        pass
    
    def add_relation(self, src_asin, dst_asin, rel_type, weight):
        # 🔧 修正：检查是否为特殊token
        if src_asin == '<PAD>' or dst_asin == '<PAD>':
            return  # 跳过特殊token
        
        # 检查索引是否为特殊token索引
        src_idx = self.item2idx.get(src_asin)
        dst_idx = self.item2idx.get(dst_asin)
        
        if (src_idx is None or dst_idx is None or 
            src_idx in self.special_token_indices or 
            dst_idx in self.special_token_indices):
            return  # 跳过特殊token或无效索引
        
        # 新增有效性断言
        assert isinstance(src_asin, str), f"无效的ASIN类型: {type(src_asin)}"
        assert isinstance(dst_asin, str), f"无效的ASIN类型: {type(dst_asin)}"
        
        # 直接存储ASIN，不进行索引转换
        key = (str(src_asin), str(dst_asin), rel_type)
        self.edge_index[key] += weight
        
    def get_edges(self):
        """获取构建的边数据"""
        return self.edge_index