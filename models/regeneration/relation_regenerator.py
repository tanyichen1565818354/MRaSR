# models/regeneration/relation_regenerator.py
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import logging
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def normal_initialization(module, initial_range=0.02):
    """DR4SR原版初始化函数"""
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

class SeqPoolingLayer(nn.Module):
    """DR4SR原版序列池化层"""
    def __init__(self, pooling_type='mean'):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, batch_seq_embeddings, seq_len):
        B, L, D = batch_seq_embeddings.shape
        mask = torch.arange(L).unsqueeze(0).unsqueeze(2).to(batch_seq_embeddings.device)
        mask = mask.expand(B, -1, D)
        seq_len = seq_len.unsqueeze(1).unsqueeze(2)
        seq_len_ = seq_len.expand(-1, mask.size(1), -1)
        mask = mask >= seq_len_
        batch_seq_embeddings = batch_seq_embeddings.masked_fill(mask, 0.0)
        
        if self.pooling_type == 'mean':
            result = batch_seq_embeddings.sum(dim=1) / (seq_len.squeeze(2) + torch.finfo(torch.float32).eps)
        elif self.pooling_type == 'sum':
            result = batch_seq_embeddings.sum(dim=1)
        else:
            result = batch_seq_embeddings.sum(dim=1) / (seq_len.squeeze(2) + torch.finfo(torch.float32).eps)
        
        return result

class RelationConditionEncoder(nn.Module):
    """关系感知条件编码器 - topK结构化关系特征"""
    def __init__(self, K, config):
        super().__init__()
        self.K = K
        self.config = config
        self.topk = 64
        
        # Transformer编码器保持不变
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=2,
            dim_feedforward=256,
            dropout=0.5,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=2,
        )
        
        # 显式关系矩阵加载
        self.explicit_relation_dict = None
        self.plm_emb = None
        self.item_mapping = None
        self._load_graph_data()
        
        # topK特征映射层
        self.co_occurrence_proj = nn.Sequential(
            nn.Linear(self.topk, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        self.sequential_proj = nn.Sequential(
            nn.Linear(self.topk, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        # PLM投影器
        self.plm_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64)
        )
        self.condition_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, K),
        )
        self.pooling_layer = SeqPoolingLayer('mean')
        self.tau = 1.0
        self.feature_mode = getattr(config.model, "relation_feature_mode", "fused")

    @staticmethod
    def _graph_filename(graph_dir, base_name):
        """根据 graph_save_dir 目录名自动判断文件名后缀。
        graph_learned/ 下的文件带 _learned 后缀（如 plm_embeddings_768dim_learned.pth），
        graph/ 下的文件不带后缀。这样切换 graph_save_dir 路径即可自动适配。
        """
        if 'learned' in Path(graph_dir).name:
            stem, ext = base_name.rsplit('.', 1)
            return f"{stem}_learned.{ext}"
        return base_name

    def _load_graph_data(self):
        """加载图谱数据，并触发关系特征查找表的预计算/加载"""
        # 初始化快速查找表
        self._co_topk_table = None
        self._seq_topk_table = None
        self._plm_table = None
        self._orig2comp_tensor = None

        try:
            graph_dir = Path(self.config.data.graph_save_dir)

            # 1. 加载PLM嵌入 (768维)
            plm_emb_path = graph_dir / self._graph_filename(graph_dir, "plm_embeddings_768dim.pth")
            if plm_emb_path.exists():
                self.plm_emb = torch.load(plm_emb_path, map_location='cpu', weights_only=False)
                logger.info(f"✅ 加载PLM嵌入: {self.plm_emb.shape} (from {plm_emb_path.name})")

            # 2. 加载显式关系矩阵
            explicit_relations_path = graph_dir / self._graph_filename(graph_dir, "explicit_relations.pth")
            if explicit_relations_path.exists():
                self.explicit_relation_dict = torch.load(explicit_relations_path, map_location='cpu', weights_only=False)
                logger.info(f"✅ 加载显式关系矩阵: {list(self.explicit_relation_dict.keys())} (from {explicit_relations_path.name})")

            # 3. 加载映射关系
            mappings_path = graph_dir / self._graph_filename(graph_dir, "mappings.pth")
            if mappings_path.exists():
                mappings = torch.load(mappings_path, map_location='cpu', weights_only=False)
                self.item_mapping = {
                    'item2idx': mappings['item2idx'],
                    'original_to_compressed': mappings.get('original_to_compressed', {}),
                    'compressed_to_original': mappings.get('compressed_to_original', {})
                }
                logger.info(f"✅ 加载映射关系: {len(self.item_mapping['item2idx'])}个物品 (from {mappings_path.name})")

            # 4. 预计算/加载 topK 特征查找表（关键加速步骤）
            # 缓存文件也带 _learned 后缀，避免与手调版 graph 的缓存冲突
            cache_path = graph_dir / self._graph_filename(graph_dir, "feature_topk_cache.pth")
            self._build_feature_cache(str(cache_path))

        except Exception as e:
            logger.warning(f"加载图谱数据失败: {e}")
            self.explicit_relation_dict = None
            self.plm_emb = None
            self.item_mapping = None

    @staticmethod
    def _sparse_to_topk_table(sparse_mat: torch.Tensor, N: int, topk: int) -> torch.Tensor:
        """
        将稀疏 COO 矩阵转为密集 topK 查找表，完全向量化，不使用切片/as_strided。
        原理：对所有非零元素按 (行, -值) 排序，计算每行内的排名，保留排名 < topk 的元素。
        """
        mat = sparse_mat.coalesce()
        if mat._nnz() == 0:
            return torch.zeros(N, topk)

        row_idx = mat.indices()[0]          # [nnz]
        vals    = mat.values().float()      # [nnz]

        # 按行升序、行内按值降序排列
        # compound key = row * scale - val，scale 须大于值域宽度
        scale = float(vals.abs().max().item() + 1.0) * 2.0
        compound = row_idx.double() * scale - vals.double()
        perm = torch.argsort(compound, stable=True)

        sorted_rows = row_idx[perm]         # [nnz] 行索引（已排序）
        sorted_vals = vals[perm]            # [nnz] 对应值（行内降序）

        # 向量化计算每元素在其行内的排名（cumcount per group）
        row_changes = torch.cat([
            torch.ones(1, dtype=torch.bool),
            sorted_rows[1:] != sorted_rows[:-1]
        ])
        group_starts = torch.where(row_changes)[0]          # [G] 每组起始位置

        global_pos  = torch.arange(mat._nnz(), dtype=torch.long)
        # searchsorted 找到每个位置属于哪个组
        group_id    = torch.searchsorted(
            group_starts.contiguous().float(),
            global_pos.float(),
            right=True
        ) - 1
        group_id    = group_id.clamp(0, len(group_starts) - 1)
        rank        = global_pos - group_starts[group_id]   # [nnz] 行内排名 0,1,2,...

        # 只保留排名 < topk 的元素，写入结果表
        mask   = rank < topk
        result = torch.zeros(N, topk)
        result[sorted_rows[mask], rank[mask]] = sorted_vals[mask]
        return result

    def _build_feature_cache(self, cache_path: str):
        """
        将稀疏关系矩阵预计算为密集 topK 查找表并缓存到磁盘。
        首次调用约需几分钟，之后直接从磁盘加载（<1秒）。
        表格形状：[N_compressed, topk]，训练时通过索引 O(1) 查表。
        """

        if Path(cache_path).exists():
            logger.info(f"加载关系特征缓存: {cache_path}")
            cache = torch.load(cache_path, map_location='cpu', weights_only=False)
            self._co_topk_table  = cache['co_topk_table']
            self._seq_topk_table = cache['seq_topk_table']
            self._plm_table      = cache.get('plm_table')
            logger.info(
                f"✅ 缓存加载完成: co={self._co_topk_table.shape}, "
                f"seq={self._seq_topk_table.shape}"
            )
            return

        if self.explicit_relation_dict is None or self.item_mapping is None:
            logger.warning("关系矩阵或映射表缺失，跳过预计算")
            return

        co_mat  = self.explicit_relation_dict.get('co_occurrence')
        seq_mat = self.explicit_relation_dict.get('sequential')
        if co_mat is None or seq_mat is None:
            logger.warning("缺少 co_occurrence 或 sequential 矩阵，跳过预计算")
            return

        N    = co_mat.shape[0]
        topk = self.topk
        logger.info(f"首次运行：预计算 topK 关系特征查找表（N={N}, topk={topk}）...")

        self._co_topk_table  = self._sparse_to_topk_table(co_mat,  N, topk).float()
        self._seq_topk_table = self._sparse_to_topk_table(seq_mat, N, topk).float()
        self._plm_table = self.plm_emb[:N].float() if self.plm_emb is not None else None

        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save({
            'co_topk_table':  self._co_topk_table,
            'seq_topk_table': self._seq_topk_table,
            'plm_table':      self._plm_table,
        }, cache_path)
        logger.info(
            f"✅ 预计算完成并缓存: {cache_path}\n"
            f"   co_topk={self._co_topk_table.shape}, seq_topk={self._seq_topk_table.shape}"
        )
        # 释放已不再需要的原始稀疏矩阵
        self.explicit_relation_dict = None

    def _compute_position_wise_weighted_average(self, avg_plm, co_emb, seq_emb):
        stacked = torch.stack([avg_plm, co_emb, seq_emb], dim=0)  # [3, 64]
        weights = F.softmax(stacked, dim=0)
        return (weights * stacked).sum(dim=0)                       # [64]

    def _ensure_tables_on_device(self, device):
        """把查找表移到目标设备（只在设备变化时执行）"""
        if self._co_topk_table is not None and self._co_topk_table.device != device:
            self._co_topk_table = self._co_topk_table.to(device)
        if self._seq_topk_table is not None and self._seq_topk_table.device != device:
            self._seq_topk_table = self._seq_topk_table.to(device)
        if self._plm_table is not None and self._plm_table.device != device:
            self._plm_table = self._plm_table.to(device)

    def _ensure_orig2comp_tensor(self, device):
        """懒加载：把 original_to_compressed 字典转为 LongTensor 方便向量化索引"""
        if self._orig2comp_tensor is not None:
            if self._orig2comp_tensor.device != device:
                self._orig2comp_tensor = self._orig2comp_tensor.to(device)
            return
        if self.item_mapping is None:
            return
        orig2comp = self.item_mapping['original_to_compressed']
        if not orig2comp:
            return
        max_orig = max(orig2comp.keys())
        table = torch.full((max_orig + 1,), -1, dtype=torch.long)
        for orig, comp in orig2comp.items():
            table[orig] = comp
        self._orig2comp_tensor = table.to(device)

    def _compute_sequence_relation_features(self, sequence_indices):
        """向量化关系特征计算：一次性查表 + 屏蔽均值，无 per-sample Python 循环"""
        batch_size, L = sequence_indices.shape
        device = sequence_indices.device

        # 查找表未就绪时返回零向量（不影响正确性，只损失关系信息）
        if self._co_topk_table is None:
            return torch.zeros(batch_size, 64, device=device)

        self._ensure_tables_on_device(device)
        self._ensure_orig2comp_tensor(device)

        if self._orig2comp_tensor is None:
            return torch.zeros(batch_size, 64, device=device)

        N_table = self._co_topk_table.shape[0]
        max_orig = self._orig2comp_tensor.shape[0] - 1

        # 1) 有效位置：非 PAD 且在 orig2comp 映射范围内  [B, L]
        valid_mask = (sequence_indices > 0) & (sequence_indices <= max_orig)
        # 安全索引到 orig2comp 表（无效位置后面用 mask 屏蔽贡献）
        safe_idx = sequence_indices.clamp(0, max_orig)
        comp_idx = self._orig2comp_tensor[safe_idx]              # [B, L]，未映射处为 -1
        comp_valid = (comp_idx >= 0) & (comp_idx < N_table)
        final_mask = valid_mask & comp_valid                     # [B, L]

        # 2) 安全索引到 topK 表（无效位置先 clamp，贡献由 mask 屏蔽）
        safe_comp = comp_idx.clamp(0, N_table - 1)
        co_vecs  = self._co_topk_table[safe_comp]                # [B, L, topk]
        seq_vecs = self._seq_topk_table[safe_comp]               # [B, L, topk]

        fm = final_mask.unsqueeze(-1).float()                    # [B, L, 1]
        cnt = final_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)   # [B, 1]
        co_mean  = (co_vecs  * fm).sum(dim=1) / cnt              # [B, topk]
        seq_mean = (seq_vecs * fm).sum(dim=1) / cnt              # [B, topk]

        co_feat  = self.co_occurrence_proj(co_mean)              # [B, 64]
        seq_feat = self.sequential_proj(seq_mean)               # [B, 64]

        # 3) PLM 特征（同样向量化屏蔽均值）
        if self._plm_table is not None:
            N_plm = self._plm_table.shape[0]
            plm_valid = final_mask & (comp_idx < N_plm)
            safe_comp_plm = comp_idx.clamp(0, N_plm - 1)
            plm_vecs = self._plm_table[safe_comp_plm]           # [B, L, 768]
            pm = plm_valid.unsqueeze(-1).float()
            plm_cnt = plm_valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
            plm_mean = (plm_vecs * pm).sum(dim=1) / plm_cnt     # [B, 768]
            plm_feat = self.plm_proj(plm_mean)                  # [B, 64]
        else:
            plm_feat = torch.zeros(batch_size, 64, device=device)

        if self.feature_mode == "plm_only":
            return plm_feat

        # 4) 三者加权融合（原 per-sample softmax 推广到 batch 维，softmax dim=0 一致）
        return self._compute_position_wise_weighted_average(plm_feat, co_feat, seq_feat)

    def forward(self, trm_input, src_mask, memory_key_padding_mask, src_seqlen, sequence_indices=None):
        # Transformer编码
        trm_out = self.encoder(
            src=trm_input,
            mask=src_mask,
            src_key_padding_mask=memory_key_padding_mask,
        )
        
        # 池化
        pooled_out = self.pooling_layer(trm_out, src_seqlen)  # [B, 64]
        
        # 🔧 新架构：动态计算关系特征
        if sequence_indices is not None:
            relation_features = self._compute_sequence_relation_features(sequence_indices)  # [B, 64]
            relation_emb = relation_features  # 直接使用64维特征，无需额外投影
            pooled_out = pooled_out + relation_emb
        
        # 生成条件
        condition = self.condition_layer(pooled_out)
        condition = F.gumbel_softmax(condition, tau=self.tau, dim=-1)
        
        self.condition4loss = condition
        self.tau = max(self.tau * 0.995, 0.1)
        
        return condition

class RelationGenerator(nn.Module):
    """关系增强生成器 - 基于DR4SR原版架构，支持多行为嵌入"""
    def __init__(self, config, num_items, sasrec_emb_path=None):
        super().__init__()
        self.config = config
        self.num_items = num_items
        self.device = config.resources.device

        # DR4SR原版参数
        self.K = config.model.K
        self.d_model = config.model.d_model
        self.max_seq_len = config.training.max_seq_len

        self.PAD = 0
        self.SOS = num_items + 1
        self.EOS = num_items + 2

        logger.info(f"特殊token定义: PAD={self.PAD}, SOS={self.SOS}, EOS={self.EOS}")
        logger.info(f"实际商品数: {num_items}, 总词汇表大小: {num_items + 3}")

        # ---- 多行为嵌入层 ----
        mb_cfg = getattr(config, "multi_behavior", None)
        self.use_behavior_emb = bool(mb_cfg and getattr(mb_cfg, "enabled", False))
        num_behaviors = int(getattr(mb_cfg, "num_behaviors", 7)) if mb_cfg else 7
        if self.use_behavior_emb:
            # Tenrec behavior ids occupy 0..num_behaviors-1 (click/like/comment/follow/share/favorite/read).
            self.behavior_embedding = nn.Embedding(num_behaviors, self.d_model)
            nn.init.normal_(self.behavior_embedding.weight, std=0.02)
            logger.info(f"✅ 已启用多行为嵌入: {num_behaviors} 种行为类型, d_model={self.d_model}")

            self.num_behaviors = num_behaviors
            # 双头 decoder 开关：默认启用；消融时在 config 设 use_behavior_head: false
            # 关闭后仍保留输入侧行为嵌入，但 item 生成退化为单头（不预测/不条件化行为）。
            self.use_behavior_head = bool(getattr(mb_cfg, "use_behavior_head", True))
            if self.use_behavior_head:
                # ── 行为条件化双头 decoder ──────────────────────────────────
                # behavior_head: 从 decoder 隐状态预测每个位置的行为类型
                # behavior_condition_proj: 将 [decoder_out || behavior_emb] 融合后投影回 d_model，
                #   使 item 生成接受行为预测的显式条件化（而非朴素双头的并行独立预测）。
                # 训练时用 ground-truth behavior 做 teacher forcing，推理时用 behavior_head 的预测。
                self.behavior_head = nn.Linear(self.d_model, num_behaviors)
                self.behavior_condition_proj = nn.Sequential(
                    nn.Linear(self.d_model * 2, self.d_model),
                    nn.ReLU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                nn.init.normal_(self.behavior_head.weight, std=0.02)
                nn.init.zeros_(self.behavior_head.bias)
                logger.info("✅ 已启用行为条件化双头 decoder (behavior_head + behavior_condition_proj)")
            else:
                self.behavior_head = None
                self.behavior_condition_proj = None
                logger.info("⚠️ 双头 decoder 已关闭（消融模式）：仅保留输入侧行为嵌入，item 生成走单头路径")
        else:
            self.behavior_embedding = None
            self.behavior_head = None
            self.behavior_condition_proj = None
            self.use_behavior_head = False
            logger.info("多行为嵌入未启用（单行为模式）")

        # DR4SR原版Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=config.model.nhead,
            num_encoder_layers=config.model.num_layers,
            num_decoder_layers=config.model.num_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
        )

        # DR4SR原版条件处理
        self.condition_linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * self.K),
            nn.ReLU(),
            nn.Linear(self.d_model * self.K, self.d_model * self.K)
        )

        self.condition_encoder = RelationConditionEncoder(self.K, config)

        # DR4SR原版其他组件
        self.dropout = nn.Dropout(config.model.dropout)
        # +2 是为 SOS/EOS 特殊token预留位置（序列张量 = [SOS] + items + [EOS]）
        self.position_embedding = nn.Embedding(config.training.max_seq_len + 2, self.d_model)

        # 初始化
        self.apply(normal_initialization)
        self.load_pretrained_embeddings(sasrec_emb_path)

    def load_pretrained_embeddings(self, sasrec_emb_path):
        """🔧 修复：完全按照DR4SR原版方式加载嵌入"""
        if sasrec_emb_path and torch.cuda.is_available():
            try:
                logger.info(f"加载SASRec预训练嵌入: {sasrec_emb_path}")
                saved = torch.load(sasrec_emb_path, map_location='cpu', weights_only=False)
                
                # 获取预训练嵌入
                if isinstance(saved, torch.Tensor):
                    pretrained = saved
                    logger.info(f"检测到直接tensor格式，形状: {pretrained.shape}")
                elif isinstance(saved, dict):
                    if 'item_embedding.weight' in saved:
                        pretrained = saved['item_embedding.weight']
                    elif 'parameters' in saved and 'item_embedding.weight' in saved['parameters']:
                        pretrained = saved['parameters']['item_embedding.weight']
                    else:
                        raise ValueError("未找到有效的嵌入tensor")
                else:
                    raise ValueError(f"未知的保存格式: {type(saved)}")
                
                logger.info(f"预训练嵌入形状: {pretrained.shape}")
                
                # 验证嵌入维度
                if pretrained.size(1) != self.d_model:
                    raise ValueError(f"嵌入维度不匹配: 预训练={pretrained.size(1)}, 模型={self.d_model}")
                
                # 🔧 关键修复：按DR4SR原版方式处理嵌入
                # 预训练嵌入应该包含 PAD + 实际物品，总共 num_items + 1 个
                expected_pretrained_size = self.num_items + 1  # PAD + 12101个实际物品 = 12102
                
                if pretrained.size(0) == expected_pretrained_size:
                    # 预训练嵌入包含PAD+物品，按DR4SR原版方式添加SOS和EOS
                    logger.info(f"预训练嵌入包含PAD+物品({expected_pretrained_size}个)，添加SOS和EOS")
                    
                    # 🔧 DR4SR原版方式：为SOS和EOS添加随机初始化嵌入
                    sos_eos_emb = torch.zeros(2, self.d_model)
                    nn.init.normal_(sos_eos_emb, std=0.02)  # 与DR4SR原版完全一致
                    
                    # 最终嵌入：[PAD(0) + Items(1-12101)] + [SOS(12102), EOS(12103)]
                    final_embeddings = torch.cat([pretrained, sos_eos_emb])
                    
                elif pretrained.size(0) == self.num_items:
                    # 预训练嵌入只包含实际物品，需要添加PAD、SOS、EOS
                    logger.info(f"预训练嵌入只包含{self.num_items}个物品，添加PAD、SOS、EOS")
                    
                    # 为特殊token创建嵌入
                    pad_emb = torch.zeros(1, self.d_model)  # PAD使用零向量
                    sos_eos_emb = torch.zeros(2, self.d_model)
                    nn.init.normal_(sos_eos_emb, std=0.02)  # SOS和EOS随机初始化
                    
                    # 按照索引顺序拼接：[PAD(0), items(1-12101), SOS(12102), EOS(12103)]
                    final_embeddings = torch.cat([pad_emb, pretrained, sos_eos_emb])
                    
                else:
                    logger.warning(f"嵌入数量不匹配: 预训练={pretrained.size(0)}, 期望={expected_pretrained_size}或{self.num_items}")
                    self._init_random_embeddings()
                    return
                
                # 创建嵌入层
                self.item_embedding = nn.Embedding.from_pretrained(
                    final_embeddings, padding_idx=self.PAD, freeze=False
                )
                self.item_embedding_decoder = self.item_embedding
                
                logger.info(f"✅ 成功加载DR4SR原版格式嵌入:")
                logger.info(f"  - 最终嵌入形状: {final_embeddings.shape}")
                logger.info(f"  - PAD: {self.PAD}")
                logger.info(f"  - 实际物品: 1-{self.num_items}")
                logger.info(f"  - SOS: {self.SOS} (随机初始化)")
                logger.info(f"  - EOS: {self.EOS} (随机初始化)")
                logger.info(f"  - 嵌入统计: 范围[{final_embeddings.min():.4f}, {final_embeddings.max():.4f}]")
                
            except Exception as e:
                logger.warning(f"加载预训练嵌入失败: {e}，使用随机初始化")
                logger.warning(f"错误详情: {str(e)}")
                self._init_random_embeddings()
        else:
            self._init_random_embeddings()

    def _init_random_embeddings(self):
        """随机初始化嵌入 - 确保包含所有特殊token"""
        total_vocab = self.num_items + 3  # PAD + 实际物品 + SOS + EOS = 12104
        self.item_embedding = nn.Embedding(total_vocab, self.d_model, padding_idx=self.PAD)
        self.item_embedding_decoder = self.item_embedding
        
        # 🔧 确保SOS和EOS有正确的随机初始化
        with torch.no_grad():
            nn.init.normal_(self.item_embedding.weight[self.SOS], std=0.02)
            nn.init.normal_(self.item_embedding.weight[self.EOS], std=0.02)
        
        logger.info(f"随机初始化嵌入，词汇表大小: {total_vocab}")
        logger.info(f"  - PAD: {self.PAD} (零初始化)")
        logger.info(f"  - 实际物品: 1-{self.num_items} (随机初始化)")
        logger.info(f"  - SOS: {self.SOS} (随机初始化)")
        logger.info(f"  - EOS: {self.EOS} (随机初始化)")

    def condition_mask(self, logits, src, tgt=None, training=True):
        """Restrict item logits to tokens that appear in src (and tgt when training)."""
        B, T, V = logits.shape
        mask = torch.zeros(B, V, dtype=torch.bool, device=logits.device)

        if training and tgt is not None:
            # 非 reduce scatter：把 1.0 写到 src/tgt 非 PAD token 对应列（重复写 1.0 幂等；
            # PAD 位置 clamp 到 col 0 写 0.0，col 0 随后被特殊 token 强制 True，故无冲突）。
            # 用普通赋值 scatter 避免 reduce='add' 的废弃告警。
            acc = torch.zeros(B, V, device=logits.device)
            acc.scatter_(1, src.clamp(min=0), (src != self.PAD).float())
            acc.scatter_(1, tgt.clamp(min=0), (tgt != self.PAD).float())
            mask = acc > 0
        else:
            # 推理时：源序列 + 基于关系的扩展（推理不在训练热路径，保留原逻辑）
            for b in range(B):
                src_tokens = src[b][src[b] != self.PAD]
                if len(src_tokens) > 0:
                    mask[b, src_tokens] = True

                    # 基于关系图谱的智能扩展
                    if hasattr(self, 'config') and self.config.model.get('enable_relation_expansion', True):
                        related_items = self._get_related_items_smart(src_tokens)
                        if len(related_items) > 0:
                            mask[b, related_items] = True

        # 特殊token总是允许
        mask[:, self.PAD] = True
        mask[:, self.SOS] = True
        mask[:, self.EOS] = True

        # masked_fill_ 原地修改：省一次 [B,T,V] 全量分配（大词表时是 GB 级）。
        # 填充值用 dtype 的最小有限值：fp16 下 -1e9 会溢出，-65504 既能屏蔽又不溢出，
        # 且 exp(-65504)=0，对 log_softmax 等价于 -inf。logits 是 matmul 的新输出，
        # 无其他消费者，原地修改对 autograd 安全（matmul 反向不保存输出，masked_fill_ 反向只需 mask）。
        fill_val = torch.finfo(logits.dtype).min
        mask_expanded = mask.unsqueeze(1).expand(-1, T, -1)
        logits.masked_fill_(~mask_expanded, fill_val)
        return logits

    def _get_related_items_smart(self, src_tokens):
        """Optional decode-mask expansion from the item relation graph.

        The sparse relation matrices are released after the top-K feature cache
        is built, so this returns an empty tensor and the mask keeps source
        tokens plus special tokens. Do not fall back to item-id proximity.
        """
        return torch.empty(0, dtype=torch.long, device=src_tokens.device)

    def _add_behavior_emb(self, item_emb, behavior_ids):
        """Add behavior embeddings to item embeddings.

        behavior_ids: [B, L] with Tenrec ids in {click=0, like=1, comment=2,
        follow=3, share=4, favorite=5, read=6}. Returns item_emb unchanged
        when multi-behavior embeddings are disabled or behavior_ids is None.
        """
        if not self.use_behavior_emb or self.behavior_embedding is None or behavior_ids is None:
            return item_emb
        # 裁剪/对齐长度
        L = item_emb.size(1)
        if behavior_ids.size(1) != L:
            if behavior_ids.size(1) > L:
                behavior_ids = behavior_ids[:, :L]
            else:
                pad = torch.zeros(
                    behavior_ids.size(0), L - behavior_ids.size(1),
                    dtype=behavior_ids.dtype, device=behavior_ids.device
                )
                behavior_ids = torch.cat([behavior_ids, pad], dim=1)
        return item_emb + self.behavior_embedding(behavior_ids)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                memory_key_padding_mask, src_seqlen, tgt_seqlen, relation_features=None,
                src_behaviors=None, tgt_behaviors=None):
        """
        前向传播。
        src_behaviors: [B, src_L] 源序列各位置的行为类型，多行为模式下传入。
        tgt_behaviors: [B, tgt_L] 目标序列各位置的行为类型，多行为模式下传入。
        """
        # 源序列编码
        position_ids = torch.arange(src.size(1), dtype=torch.long, device=self.device).reshape(1, -1)
        src_position_embedding = self.position_embedding(position_ids)
        src_item_emb = self.item_embedding(src) + src_position_embedding
        # 叠加行为嵌入
        src_item_emb = self._add_behavior_emb(src_item_emb, src_behaviors)
        src_emb = self.dropout(src_item_emb)

        memory = self.transformer.encoder(src_emb, src_mask, src_padding_mask)
        B, L, D = memory.shape
        memory = self.condition_linear(memory).reshape(B, L, self.K, D)

        # 目标序列编码
        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device).reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        tgt_item_emb = self.item_embedding(tgt) + tgt_position_embedding
        tgt_item_emb = self._add_behavior_emb(tgt_item_emb, tgt_behaviors)
        tgt_emb = self.dropout(tgt_item_emb)

        condition = self.condition_encoder(
            tgt_emb, tgt_mask, tgt_padding_mask, tgt_seqlen, sequence_indices=src
        )  # [B, K]
        condition = condition.reshape(B, 1, self.K, 1)
        memory_cond = (memory * condition).sum(-2)

        outs = self.transformer.decoder(
            tgt_emb, memory_cond, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask
        )

        # ── 行为条件化双头 decoder ──────────────────────────────────────
        if self.use_behavior_head:
            # 1) behavior_head 先出：从 decoder 隐状态预测每个位置的行为
            behavior_logits = self.behavior_head(outs)  # [B, L, num_behaviors]

            # 2) 用行为条件化 item 生成
            if self.training and tgt_behaviors is not None:
                # Teacher forcing：训练时用 ground-truth behavior 做 condition
                beh_for_cond = tgt_behaviors
            else:
                # 推理时用 behavior_head 自己的预测
                beh_for_cond = behavior_logits.argmax(-1)

            beh_emb = self.behavior_embedding(beh_for_cond)  # [B, L, D]
            conditioned = self.behavior_condition_proj(
                torch.cat([outs, beh_emb], dim=-1)
            )  # [B, L, D]
            logits = conditioned @ self.item_embedding_decoder.weight.T

            # condition_mask 只作用于 item logits，不影响 behavior_logits
            if self.training:
                full_tgt = torch.cat([tgt, torch.zeros(tgt.size(0), 1, dtype=tgt.dtype, device=tgt.device)], dim=1)
                logits = self.condition_mask(logits, src, full_tgt, training=True)
            else:
                logits = self.condition_mask(logits, src, None, training=False)

            return logits, behavior_logits

        # ── 单行为路径（原逻辑不变）──────────────────────────────────────
        logits = outs @ self.item_embedding_decoder.weight.T

        if self.training:
            full_tgt = torch.cat([tgt, torch.zeros(tgt.size(0), 1, dtype=tgt.dtype, device=tgt.device)], dim=1)
            logits = self.condition_mask(logits, src, full_tgt, training=True)
        else:
            logits = self.condition_mask(logits, src, None, training=False)

        return logits, None

    def encode(self, src, src_mask, src_behaviors=None):
        """编码（支持多行为嵌入）"""
        position_ids = torch.arange(src.size(1), dtype=torch.long, device=self.device).reshape(1, -1)
        src_position_embedding = self.position_embedding(position_ids)
        src_item_emb = self.item_embedding(src) + src_position_embedding
        src_item_emb = self._add_behavior_emb(src_item_emb, src_behaviors)
        src_emb = self.dropout(src_item_emb)
        return self.transformer.encoder(src_emb, src_mask)

    def set_condition(self, condition):
        """设置条件 - DR4SR原版"""
        self.condition = condition

    def decode(self, tgt, memory, tgt_mask):
        """解码 - 返回 (decoder_out, behavior_logits)。
        behavior_logits 在单行为模式下为 None。"""
        B, L, D = memory.shape
        memory = self.condition_linear(memory).reshape(B, L, self.K, D)[:, :, self.condition]
        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        tgt_emb = self.dropout(self.item_embedding(tgt) + tgt_position_embedding)
        decoder_out = self.transformer.decoder(tgt_emb, memory, tgt_mask)

        if self.use_behavior_head:
            behavior_logits = self.behavior_head(decoder_out)  # [B, L, num_behaviors]
            return decoder_out, behavior_logits
        return decoder_out, None

    def compute_item_logits_from_decoder_out(self, decoder_out, behavior_logits):
        """从 decoder 输出计算 item logits，支持行为条件化。

        推理时逐步调用：decoder_out [B, L, D], behavior_logits [B, L, num_beh] or None。
        返回 [B, L, vocab_size] 的 item logits，取最后位置即可。
        """
        if self.use_behavior_head and behavior_logits is not None:
            # 用 behavior_head 的预测条件化 item logits
            predicted_beh = behavior_logits.argmax(-1)  # [B, L]
            beh_emb = self.behavior_embedding(predicted_beh)  # [B, L, D]
            conditioned = self.behavior_condition_proj(
                torch.cat([decoder_out, beh_emb], dim=-1)
            )  # [B, L, D]
            return conditioned @ self.item_embedding_decoder.weight.T
        return decoder_out @ self.item_embedding_decoder.weight.T

def generate_square_subsequent_mask(sz, device='cuda'):
    """DR4SR原版掩码生成"""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, -100000).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    """DR4SR原版掩码创建"""
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    
    device = src.device
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = generate_square_subsequent_mask(src_seq_len, device)
    
    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def dr4sr_cross_entropy_loss(logits, targets, ignore_index=0, chunk_rows=0):
    """DR4SR交叉熵损失函数。

    chunk_rows > 0 时按行分块计算：每块单独 upcast 到 fp32 做 CE（reduction='sum'），
    最后除以有效 token 数得到与全局 mean 等价的结果。
    这样避免一次性把 [N, V] 的 fp32 logits 物化出来——大词表（V≈36万）时
    全量 .float() 单张就 4GB+，是 OOM 的主因。分块后峰值降到 [chunk_rows, V]。
    chunk_rows <= 0 时走原版逻辑（小词表或兼容老调用方）。
    """
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    N = logits_flat.shape[0]

    if chunk_rows <= 0 or N <= chunk_rows:
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        return loss_fn(logits_flat.float(), targets_flat)

    total_loss = logits_flat.new_zeros((), dtype=torch.float32)
    total_valid = targets_flat.new_zeros((), dtype=torch.long)
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        l = F.cross_entropy(
            logits_flat[s:e].float(),
            targets_flat[s:e],
            ignore_index=ignore_index,
            reduction='sum',
        )
        total_loss = total_loss + l
        total_valid = total_valid + (targets_flat[s:e] != ignore_index).sum()
    return total_loss / total_valid.clamp(min=1).float()

def compute_dr4sr_regularization_loss(condition_prob):
    """DR4SR原版正则化损失 - 条件多样性损失"""
    # 计算条件分布的熵损失，鼓励多样性
    reg_loss = -(condition_prob * torch.log(condition_prob + 1e-12)).sum(-1).mean()
    return reg_loss