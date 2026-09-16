#!/usr/bin/env python3
"""
关系图谱构建脚本
用于构建商品关系图谱，包括显式关系和PLM相似度矩阵
"""

import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
import logging
from tqdm import tqdm
import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from models.augmentation.graph import RelationGraphBuilder

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resolve_config_path(path_value):
    """将相对路径转换为相对仓库根目录的绝对路径。"""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def normalize_config_paths(config):
    """把配置中的相对数据路径统一改为仓库根目录下的绝对路径。"""
    for key in [
        "train_dir",
        "preprocessed_dir",
        "embeddings_dir",
        "item2idx_path",
        "sequences_path",
        "products_path",
        "graph_save_dir",
        "plm_emb_path",
        "plm_id2idx_path",
    ]:
        if hasattr(config.data, key):
            raw_value = getattr(config.data, key)
            if raw_value:
                setattr(config.data, key, str(resolve_config_path(raw_value)))


def load_preprocessed_data(config):
    """加载预处理后的数据"""
    logger.info("加载预处理数据...")
    
    # 数据路径
    sequences_path = Path(config.data.sequences_path)
    products_path = Path(config.data.products_path)
    item2idx_path = Path(config.data.item2idx_path)
    
    # 检查文件是否存在
    for path in [sequences_path, products_path, item2idx_path]:
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
    
    # 加载数据
    sequences = torch.load(sequences_path, weights_only=False)
    products = torch.load(products_path, weights_only=False)
    item2idx = torch.load(item2idx_path, weights_only=False)
    
    logger.info(f"数据加载完成:")
    logger.info(f"   序列数: {len(sequences)}")
    logger.info(f"   商品数: {len(products)}")
    logger.info(f"   索引映射数: {len(item2idx)}")
    
    return sequences, products, item2idx

@hydra.main(version_base=None, config_path="../configs", config_name="graph")
def main(cfg: DictConfig) -> None:
    """主函数：构建关系图谱"""
    logger.info("开始构建关系图谱...")
    logger.info(f"使用设备: {cfg.device}")

    normalize_config_paths(cfg)
    
    try:
        # 加载预处理数据
        sequences, products, item2idx = load_preprocessed_data(cfg)
        
        # 创建图谱构建器
        logger.info("初始化图谱构建器...")
        graph_builder = RelationGraphBuilder(
            products=products,
            sequences=sequences,
            item2idx=item2idx,
            config=cfg
        )
        
        # 构建完整图谱
        logger.info("构建完整关系图谱...")
        graph_builder.build_full_graph()
        
        # 保存图谱
        logger.info("保存图谱数据...")
        graph_builder.save_graph()
        
        logger.info("关系图谱构建完成!")
        
        # 输出最终统计
        logger.info("=== 最终统计 ===")
        if hasattr(graph_builder, 'explicit_relation_dict'):
            for rel_type, matrix in graph_builder.explicit_relation_dict.items():
                logger.info(f"{rel_type}: {matrix._nnz()}个关系")
        
        if hasattr(graph_builder, 'plm_sim_matrix') and graph_builder.plm_sim_matrix is not None:
            logger.info(f"PLM相似度: {graph_builder.plm_sim_matrix.nnz()}个相似对")
        
    except Exception as e:
        logger.error(f"图谱构建失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()