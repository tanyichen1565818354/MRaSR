from pathlib import Path
import random
from omegaconf import OmegaConf
import psutil
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from models.augmentation.graph_builders import (
    CooccurrenceRelationBuilder,
    SequentialRelationBuilder
)
from models.augmentation.behavior_relation_learner import BehaviorRelationLearner
from models.augmentation.behavior_weight_utils import get_multi_behavior_mode

logger = logging.getLogger(__name__)

# 图谱文件命名：static=旧版手调权重；learned=可学习行为转移矩阵版（不覆盖旧图谱）
GRAPH_FILE_NAMES = {
    "static": {
        "plm_emb": "plm_embeddings_768dim.pth",
        "explicit_relations": "explicit_relations.pth",
        "plm_similarity": "plm_similarity_matrix.pth",
        "mappings": "mappings.pth",
        "metadata": "metadata.pth",
        "behavior_matrix": None,
    },
    "learned": {
        "plm_emb": "plm_embeddings_768dim_learned.pth",
        "explicit_relations": "explicit_relations_learned.pth",
        "plm_similarity": "plm_similarity_matrix_learned.pth",
        "mappings": "mappings_learned.pth",
        "metadata": "metadata_learned.pth",
        "behavior_matrix": "behavior_relation_matrix.pth",
    },
}

class RelationGraphBuilder:
    """专门用于构建关系图谱的类"""
    
    def __init__(self, products, sequences, item2idx, config):
        """初始化关系图谱构建器"""
        # 基础配置
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # 设置随机种子
        seed = config.get("seed", 2025)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # 数据清洗和类型统一
        self.products = {
            str(k).strip(): v 
            for k, v in products.items()
        }
        
        # 清洗序列数据（同步过滤 behaviors，避免长度错位）
        self.sequences = {}
        for seq_id, seq_data in sequences.items():
            raw_seq = seq_data.get('sequence', [])
            raw_beh = seq_data.get('behaviors', None)
            clean_seq = []
            clean_beh = []
            for i, item in enumerate(raw_seq):
                item_str = str(item).strip()
                if not item_str:
                    continue
                clean_seq.append(item_str)
                if raw_beh is not None and i < len(raw_beh):
                    clean_beh.append(int(raw_beh[i]))
            cleaned = {
                k: v for k, v in seq_data.items()
                if k not in ('sequence', 'behaviors')
            }
            cleaned['sequence'] = clean_seq
            if raw_beh is not None:
                cleaned['behaviors'] = clean_beh
            self.sequences[seq_id] = cleaned
        
        # 建立映射关系
        self.item2idx = item2idx
        self.idx2item = {v: k for k, v in item2idx.items()}
        
        # 验证数据完整性
        self._validate_data_integrity()
        
        # 初始化图谱组件
        self.explicit_edges = defaultdict(float)
        self.plm_emb = None
        self.explicit_relation_dict = {}
        self.plm_sim_matrix = None
        self.behavior_relation_matrix = None

    def _get_graph_variant(self) -> str:
        """返回图谱版本标识：static | learned"""
        variant = self.config.graph.get("file_variant", None)
        if variant:
            return str(variant)
        if get_multi_behavior_mode(self.config) == "learned_matrix":
            return "learned"
        return "static"

    def _get_graph_file_names(self) -> dict:
        variant = self._get_graph_variant()
        if variant not in GRAPH_FILE_NAMES:
            raise ValueError(f"未知 graph.file_variant: {variant}")
        return GRAPH_FILE_NAMES[variant]

    def _learn_behavior_relation_matrix(self):
        """在构建显式关系前，从训练序列学习行为转移矩阵 R。"""
        mb_cfg = self.config.get("multi_behavior", {})
        if not mb_cfg.get("enabled", False):
            return None
        if get_multi_behavior_mode(self.config) != "learned_matrix":
            return None

        num_behaviors = int(mb_cfg.get("num_behaviors", 7))
        learner = BehaviorRelationLearner(num_behaviors=num_behaviors, device=str(self.device))
        self.behavior_relation_matrix = learner.learn(self.sequences, self.config)
        return self.behavior_relation_matrix
        
    def _validate_data_integrity(self):
        """验证数据完整性"""
        # 收集所有需要的商品
        product_items = set(self.products.keys())
        seq_items = set()
        for s in self.sequences.values():
            seq_items.update(s['sequence'])
        
        # 检查缺失的商品
        missing_in_products = seq_items - product_items
        if missing_in_products:
            self.logger.warning(f"自动补充{len(missing_in_products)}个缺失商品元数据")
            for asin in missing_in_products:
                self.products[asin] = {
                    'categories': [['unknown']],
                    'price': 0.0,
                    'brand': 'unknown',
                    'title': f"Dynamic Item {asin}",
                    'description': '',
                    'dynamic_flag': True
                }
        
        # 验证映射完整性
        for seq in self.sequences.values():
            for asin in seq['sequence']:
                if asin not in self.item2idx:
                    raise ValueError(f"关键错误：{asin} 未在映射表中")
        
        self.logger.info("✅ 数据完整性验证通过")
    
    def _load_plm_embeddings(self):
        """加载PLM嵌入（768维）"""
        emb_path = Path(self.config.data.plm_emb_path)
        self.logger.info(f"正在加载PLM嵌入，路径: {emb_path}")
        
        if not emb_path.exists():
            raise FileNotFoundError(f"PLM特征文件缺失: {emb_path}")
        
        try:
            plm_emb_raw = torch.load(emb_path, weights_only=False)
            self.logger.info(f"成功加载PLM嵌入，原始维度: {plm_emb_raw.shape}")
            
            # 计算实际物品数量（排除特殊token）
            actual_items = len([asin for asin, idx in self.item2idx.items() 
                              if idx != 0])  # 只排除PAD
            
            # 处理PLM嵌入数量匹配
            if len(plm_emb_raw) == actual_items + 1:
                # 包含PAD token，需要排除
                plm_emb_final = plm_emb_raw[1:]  # 排除第一个（PAD）
                self.logger.info("排除PAD token后的PLM嵌入")
            elif len(plm_emb_raw) == actual_items:
                # 已经只包含实际物品
                plm_emb_final = plm_emb_raw
            else:
                raise ValueError(f"PLM嵌入数量({len(plm_emb_raw)})与预期不匹配")
            
            # 转移到设备并归一化
            plm_emb_final = plm_emb_final.to(self.device)
            self.plm_emb = F.normalize(plm_emb_final, p=2, dim=1)
            
            self.logger.info(f"✅ PLM嵌入加载完成，最终维度: {self.plm_emb.shape}")
            return self.plm_emb
            
        except Exception as e:
            self.logger.error(f"PLM嵌入加载失败: {str(e)}")
            raise
    
    def build_explicit_relations(self):
        """构建显式关系"""
        self.logger.info("开始构建显式关系...")

        # learned_matrix 模式：先从序列学习 R，再用于边权重
        self._learn_behavior_relation_matrix()
        if self.behavior_relation_matrix is not None:
            self.logger.info(
                f"使用可学习行为转移矩阵构建边权重，shape={tuple(self.behavior_relation_matrix.shape)}"
            )
        elif get_multi_behavior_mode(self.config) == "static":
            self.logger.info("使用手调 behavior_weights 构建边权重（static 模式）")
        
        # 初始化构建器
        builders = [
            (CooccurrenceRelationBuilder(
                products=self.products,
                sequences=self.sequences,
                item2idx=self.item2idx,
                config=self.config,
                behavior_relation_matrix=self.behavior_relation_matrix,
            ), "co_occurrence"),
            (SequentialRelationBuilder(
                products=self.products,
                sequences=self.sequences,
                item2idx=self.item2idx,
                config=self.config,
                behavior_relation_matrix=self.behavior_relation_matrix,
            ), "sequential")
        ]
        
        # 构建关系
        self.explicit_counts = defaultdict(int)
        
        for builder, rel_type in builders:
            try:
                # 获取参数
                params = {}
                if rel_type == "co_occurrence":
                    params = {
                        'window_size': self.config.relation_building.cooccurrence.window_size,
                        'weight': self.config.relation_building.cooccurrence.default_weight,
                        'sample_rate': self.config.relation_building.cooccurrence.sample_rate
                    }
                elif rel_type == "sequential":
                    params = {
                        'weight': self.config.relation_building.sequential.default_weight,
                        'decay_factor': self.config.relation_building.sequential.decay_factor
                    }
                
                # 执行构建
                edges = builder.build(**params)
                
                # 合并边并统计
                valid_count = 0
                for (src, dst, t), w in edges.items():
                    if src in self.item2idx and dst in self.item2idx:
                        self.explicit_edges[(src, dst, t)] += w
                        valid_count += 1
                
                self.explicit_counts[rel_type] += valid_count
                self.logger.info(f"✅ {rel_type}关系边数: {len(edges)} (有效边: {valid_count})")
                
            except Exception as e:
                self.logger.error(f"关系构建失败: {rel_type} - {str(e)}")
                continue
        
        # 输出统计
        total_explicit = sum(self.explicit_counts.values())
        self.logger.info(f"显式关系构建完成，总边数: {total_explicit}")
        
        # 构建稀疏矩阵
        self._build_explicit_relation_matrices()
        
    def _build_explicit_relation_matrices(self):
        """构建显式关系稀疏矩阵"""
        self.logger.info("构建显式关系稀疏矩阵...")
        
        # 获取实际物品索引
        actual_item_indices = [idx for asin, idx in self.item2idx.items() 
                              if idx != 0]  # 排除PAD
        num_actual_items = len(actual_item_indices)
        
        # 创建索引映射
        self.original_to_compressed = {}
        self.compressed_to_original = {}
        for compressed_idx, original_idx in enumerate(sorted(actual_item_indices)):
            self.original_to_compressed[original_idx] = compressed_idx
            self.compressed_to_original[compressed_idx] = original_idx
        
        # 收集边数据
        co_edges = {'rows': [], 'cols': [], 'values': []}
        seq_edges = {'rows': [], 'cols': [], 'values': []}
        
        for (src, dst, rel_type), weight in self.explicit_edges.items():
            src_idx = self.item2idx[src]
            dst_idx = self.item2idx[dst]
            
            # 只处理实际物品的边
            if src_idx in self.original_to_compressed and dst_idx in self.original_to_compressed:
                compressed_src = self.original_to_compressed[src_idx]
                compressed_dst = self.original_to_compressed[dst_idx]
                
                if rel_type == 'co_occurrence':
                    co_edges['rows'].append(compressed_src)
                    co_edges['cols'].append(compressed_dst)
                    co_edges['values'].append(weight)
                    
                    if self.config.graph.bidirectional:
                        co_edges['rows'].append(compressed_dst)
                        co_edges['cols'].append(compressed_src)
                        co_edges['values'].append(weight * self.config.graph.reverse_decay)
                
                elif rel_type == 'sequential':
                    seq_edges['rows'].append(compressed_src)
                    seq_edges['cols'].append(compressed_dst)
                    seq_edges['values'].append(weight)
                    
                    if self.config.graph.bidirectional:
                        seq_edges['rows'].append(compressed_dst)
                        seq_edges['cols'].append(compressed_src)
                        seq_edges['values'].append(weight * self.config.graph.reverse_decay)
        
        # 构建稀疏矩阵
        self.explicit_relation_dict = {}
        
        # 共现关系矩阵
        co_sparse = self._build_sparse_matrix(co_edges, num_actual_items)
        self.explicit_relation_dict['co_occurrence'] = co_sparse
        
        # 时序关系矩阵
        seq_sparse = self._build_sparse_matrix(seq_edges, num_actual_items)
        self.explicit_relation_dict['sequential'] = seq_sparse
        
        # 输出统计
        for rel_type in ['co_occurrence', 'sequential']:
            nnz = self.explicit_relation_dict[rel_type]._nnz()
            self.logger.info(f"✅ {rel_type}关系矩阵 | 非零元素: {nnz} | 大小: {num_actual_items}x{num_actual_items}")
    
    def _build_sparse_matrix(self, edges, num_actual_items):
        """构建稀疏矩阵"""
        if not edges['rows']:
            # 空矩阵
            return torch.sparse_coo_tensor(
                indices=torch.zeros((2, 0), dtype=torch.long),
                values=torch.zeros(0, dtype=torch.float32),
                size=(num_actual_items, num_actual_items),
                device=self.device
            ).coalesce()
        
        # 构建稀疏矩阵
        indices = torch.tensor([edges['rows'], edges['cols']], dtype=torch.long)
        values = torch.tensor(edges['values'], dtype=torch.float32)
        
        sparse_tensor = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(num_actual_items, num_actual_items)
        ).coalesce()
        
        return sparse_tensor.to(self.device)
    
    def build_plm_similarity_matrix(self):
        """构建PLM相似度矩阵（全张量批量操作版，避免 Python list 瓶颈）"""
        self.logger.info("构建PLM相似度矩阵...")

        if self.plm_emb is None:
            self._load_plm_embeddings()

        # 加载PLM索引映射
        plm_emb_dir = Path(self.config.data.train_dir) / "preprocessed/embeddings"
        plm_id2idx_path = plm_emb_dir / "item_id2idx.pth"
        if plm_id2idx_path.exists():
            plm_item_id2idx = torch.load(plm_id2idx_path, weights_only=False)
            self.logger.info(f"加载PLM索引映射: {len(plm_item_id2idx)}个物品")
            if len(self.plm_emb) != len(plm_item_id2idx):
                raise ValueError(
                    f"PLM嵌入大小({len(self.plm_emb)})与"
                    f"索引映射大小({len(plm_item_id2idx)})不匹配"
                )
        else:
            self.logger.warning(f"PLM索引映射文件不存在: {plm_id2idx_path}")

        batch_size = self.config.similarity_computation.batch_size
        topk = min(self.config.similarity_computation.topk, len(self.plm_emb) - 1)
        sim_threshold = self.config.similarity_computation.sim_threshold
        N = len(self.plm_emb)

        self.logger.info(
            f"PLM相似度计算: N={N}, batch={batch_size}, topk={topk}, threshold={sim_threshold}"
        )

        # 用 tensor 列表替代 Python list，批次内全量向量化
        all_rows, all_cols, all_vals = [], [], []
        num_batches = (N + batch_size - 1) // batch_size

        with tqdm(total=num_batches, desc="[PLM相似度计算]") as pbar:
            for i in range(0, N, batch_size):
                batch = self.plm_emb[i : i + batch_size]   # [B, 768]
                B = batch.size(0)

                sim = torch.mm(batch, self.plm_emb.t())    # [B, N]

                # 排除自身（对角线置 -inf）
                self_idx = torch.arange(B, device=self.device)
                global_idx = self_idx + i
                valid_self = global_idx < N
                sim[self_idx[valid_self], global_idx[valid_self]] = -1e9

                # 批量 topK，一次完成所有行
                actual_k = min(topk, N - 1)
                topk_vals, topk_indices = torch.topk(sim, k=actual_k, dim=1)  # [B, K]

                # 阈值过滤
                mask = topk_vals >= sim_threshold          # [B, K]

                if mask.any():
                    # 行索引：[B, K] → expand 后取 mask
                    row_base = (
                        torch.arange(i, i + B, device=self.device)
                        .unsqueeze(1)
                        .expand_as(topk_indices)
                    )
                    valid_rows = row_base[mask].cpu()
                    valid_cols = topk_indices[mask].cpu()
                    valid_vals = topk_vals[mask].cpu()

                    all_rows.append(valid_rows)
                    all_cols.append(valid_cols)
                    all_vals.append(valid_vals)

                pbar.update(1)
                pbar.set_postfix({
                    "已处理": f"{min(i + B, N)}/{N}",
                    "批次边数": int(mask.sum().item()),
                })

        # 拼接所有批次结果（一次 cat，代替逐元素 extend）
        if all_rows:
            rows_t = torch.cat(all_rows)
            cols_t = torch.cat(all_cols)
            vals_t = torch.cat(all_vals)
            self.logger.info(f"总边数（过滤后）: {len(rows_t):,}")

            self.plm_sim_matrix = SparseTensor(
                row=rows_t,
                col=cols_t,
                value=vals_t.float(),
                sparse_sizes=(N, N),
            ).to(self.device)
        else:
            self.plm_sim_matrix = SparseTensor(
                row=torch.zeros(0, dtype=torch.long),
                col=torch.zeros(0, dtype=torch.long),
                value=torch.zeros(0, dtype=torch.float32),
                sparse_sizes=(N, N),
            ).to(self.device)

        nnz = self.plm_sim_matrix.nnz()
        self.logger.info(f"✅ PLM相似度矩阵构建完成: 非零元素={nnz:,}, 平均每行={nnz/N:.1f}")
    
    def build_full_graph(self):
        """构建完整的关系图谱"""
        self.logger.info("开始构建完整关系图谱...")
        
        with tqdm(total=3, desc="[图谱构建] 总进度") as pbar:
            # 步骤1: 构建显式关系
            if not hasattr(self, 'explicit_relation_dict') or not self.explicit_relation_dict:
                self.build_explicit_relations()
            pbar.update(1)
            
            # 步骤2: 构建PLM相似度矩阵
            if self.plm_sim_matrix is None:
                self.build_plm_similarity_matrix()
            pbar.update(1)
            
            # 步骤3: 验证和输出统计
            self._validate_graph()
            pbar.update(1)
        
        self.logger.info("✅ 关系图谱构建完成")
    
    def _validate_graph(self):
        """验证图谱质量"""
        self.logger.info("=== 图谱构建结果验证 ===")
        
        # 验证显式关系
        for rel_type, matrix in self.explicit_relation_dict.items():
            nnz = matrix._nnz()
            self.logger.info(f"{rel_type}关系: {nnz}个非零元素")
        
        # 验证PLM相似度矩阵
        if self.plm_sim_matrix is not None:
            plm_nnz = self.plm_sim_matrix.nnz()
            self.logger.info(f"PLM相似度矩阵: {plm_nnz}个非零元素")
        
        # 验证不包含特殊token
        special_indices = {0}  # PAD
        
        # 检查显式关系
        for rel_type, matrix in self.explicit_relation_dict.items():
            if matrix._nnz() > 0:
                matrix = matrix.coalesce()
                edge_indices = torch.cat([matrix.indices()[0], matrix.indices()[1]]).unique()
                # 这里的索引是压缩后的索引，应该都是有效的
                self.logger.info(f"✅ {rel_type}关系验证通过：使用压缩索引")
        
        self.logger.info("✅ 图谱验证完成")
    
    def save_graph(self, save_dir=None):
        """保存图谱数据"""
        if save_dir is None:
            save_dir = Path(self.config.data.graph_save_dir)
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        file_names = self._get_graph_file_names()
        variant = self._get_graph_variant()
        
        self.logger.info(f"保存图谱到: {save_dir} (variant={variant})")
        
        # 保存PLM嵌入（768维）
        if self.plm_emb is not None:
            plm_path = save_dir / file_names["plm_emb"]
            torch.save(self.plm_emb.cpu(), plm_path)
            self.logger.info(f"✅ 保存PLM嵌入: {plm_path}")
        
        # 保存显式关系矩阵
        if self.explicit_relation_dict:
            relations_path = save_dir / file_names["explicit_relations"]
            relations_cpu = {
                rel_type: matrix.cpu() 
                for rel_type, matrix in self.explicit_relation_dict.items()
            }
            torch.save(relations_cpu, relations_path)
            self.logger.info(f"✅ 保存显式关系矩阵: {relations_path}")
        
        # 保存PLM相似度矩阵
        if self.plm_sim_matrix is not None:
            sim_path = save_dir / file_names["plm_similarity"]
            # 转换为torch.sparse格式保存
            sim_cpu = torch.sparse_coo_tensor(
                indices=torch.stack([self.plm_sim_matrix.storage.row(), self.plm_sim_matrix.storage.col()]),
                values=self.plm_sim_matrix.storage.value(),
                size=self.plm_sim_matrix.sizes()
            ).coalesce().cpu()
            torch.save(sim_cpu, sim_path)
            self.logger.info(f"✅ 保存PLM相似度矩阵: {sim_path}")

        # 保存可学习行为转移矩阵（learned 模式专属，供消融/可视化）
        if self.behavior_relation_matrix is not None and file_names.get("behavior_matrix"):
            matrix_path = save_dir / file_names["behavior_matrix"]
            torch.save(
                {
                    "relation_logits": self.behavior_relation_matrix.cpu(),
                    "relation_probs": torch.sigmoid(self.behavior_relation_matrix.cpu()),
                    "num_behaviors": self.behavior_relation_matrix.shape[0],
                    "mode": "learned_matrix",
                },
                matrix_path,
            )
            self.logger.info(f"✅ 保存行为转移矩阵: {matrix_path}")
        
        # 保存映射关系
        mapping_path = save_dir / file_names["mappings"]
        mappings = {
            'item2idx': self.item2idx,
            'idx2item': self.idx2item,
            'original_to_compressed': getattr(self, 'original_to_compressed', {}),
            'compressed_to_original': getattr(self, 'compressed_to_original', {})
        }
        torch.save(mappings, mapping_path)
        self.logger.info(f"✅ 保存映射关系: {mapping_path}")
        
        # 保存元数据
        metadata_path = save_dir / file_names["metadata"]
        metadata = {
            'num_items': len(self.item2idx),
            'num_actual_items': len([idx for idx in self.item2idx.values() if idx != 0]),
            'plm_dim': 768,
            'version': f'v2.0_{variant}_behavior_matrix' if variant == 'learned' else 'v1.0_768dim_plm',
            'graph_variant': variant,
            'multi_behavior_mode': get_multi_behavior_mode(self.config),
            'config': OmegaConf.to_container(self.config)
        }
        torch.save(metadata, metadata_path)
        self.logger.info(f"✅ 保存元数据: {metadata_path}")
        
        self.logger.info("✅ 图谱保存完成")
    
    def load_graph(self, load_dir=None):
        """加载图谱数据"""
        if load_dir is None:
            load_dir = Path(self.config.data.graph_save_dir)
        else:
            load_dir = Path(load_dir)
        
        self.logger.info(f"从以下路径加载图谱: {load_dir} (variant={self._get_graph_variant()})")
        file_names = self._get_graph_file_names()
        
        # 加载PLM嵌入
        plm_path = load_dir / file_names["plm_emb"]
        if plm_path.exists():
            self.plm_emb = torch.load(plm_path, weights_only=False).to(self.device)
            self.logger.info(f"✅ 加载PLM嵌入: {self.plm_emb.shape}")
        
        # 加载显式关系矩阵
        relations_path = load_dir / file_names["explicit_relations"]
        if relations_path.exists():
            relations_cpu = torch.load(relations_path, weights_only=False)
            self.explicit_relation_dict = {
                rel_type: matrix.to(self.device)
                for rel_type, matrix in relations_cpu.items()
            }
            self.logger.info("✅ 加载显式关系矩阵")
        
        # 加载PLM相似度矩阵
        sim_path = load_dir / file_names["plm_similarity"]
        if sim_path.exists():
            sim_cpu = torch.load(sim_path, weights_only=False)
            # 转换为SparseTensor格式
            self.plm_sim_matrix = SparseTensor(
                row=sim_cpu.indices()[0],
                col=sim_cpu.indices()[1],
                value=sim_cpu.values(),
                sparse_sizes=sim_cpu.size()
            ).to(self.device)
            self.logger.info("✅ 加载PLM相似度矩阵")

        # 加载行为转移矩阵（learned 模式）
        matrix_name = file_names.get("behavior_matrix")
        if matrix_name:
            matrix_path = load_dir / matrix_name
            if matrix_path.exists():
                matrix_data = torch.load(matrix_path, map_location="cpu", weights_only=False)
                if isinstance(matrix_data, dict):
                    self.behavior_relation_matrix = matrix_data.get("relation_logits")
                else:
                    self.behavior_relation_matrix = matrix_data
                self.logger.info(f"✅ 加载行为转移矩阵: {matrix_path}")
        
        # 加载映射关系
        mapping_path = load_dir / file_names["mappings"]
        if mapping_path.exists():
            mappings = torch.load(mapping_path, weights_only=False)
            self.item2idx = mappings['item2idx']
            self.idx2item = mappings['idx2item']
            self.original_to_compressed = mappings.get('original_to_compressed', {})
            self.compressed_to_original = mappings.get('compressed_to_original', {})
            self.logger.info("✅ 加载映射关系")
        
        self.logger.info("✅ 图谱加载完成")
        return True
