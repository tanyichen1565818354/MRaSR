"""
完整版数据增强运行脚本
"""

import os
import sys
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict
import gc
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from models.augmentation.relation_augmenter import RelationAugmenter

# 设置日志 - 调整级别以减少输出
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置特定模块的日志级别
logging.getLogger('models.augmentation.relation_augmenter').setLevel(logging.INFO)

def load_data(config):
    """加载数据"""
    logger.info("加载数据...")
    
    # 加载商品数据
    products_path = Path(config.data.preprocessed_dir) / "products.pth"
    if not products_path.exists():
        raise FileNotFoundError(f"商品数据不存在: {products_path}")
    products = torch.load(products_path, weights_only=False)
    
    # 加载序列数据
    sequences_path = Path(config.data.preprocessed_dir) / "sequences.pth"
    if not sequences_path.exists():
        raise FileNotFoundError(f"序列数据不存在: {sequences_path}")
    sequences = torch.load(sequences_path, weights_only=False)
    
    logger.info(f"✅ 加载完成: {len(products)}个商品, {len(sequences)}个序列")
    
    # 统计序列长度分布
    length_stats = defaultdict(int)
    for seq_data in sequences.values():
        length = len(seq_data['sequence'])
        length_stats[length] += 1
    
    logger.info("序列长度分布:")
    for length in sorted(length_stats.keys())[:10]:  # 显示前10个长度
        logger.info(f"  长度 {length}: {length_stats[length]} 个序列")
    
    return products, sequences

def monitor_memory():
    """监控内存使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU内存: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")

@hydra.main(version_base=None, config_path="../configs", config_name="augmentation")
def main(cfg: DictConfig) -> None:
    """主函数"""
    logger.info("=== 启动完整版数据增强 ===")
    
    # 设置随机种子
    seed = cfg.get("seed", 2025)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 显示配置信息
    logger.info(f"随机种子: {seed}")
    logger.info(f"目标生成数量: {cfg.target_count}")
    logger.info(f"设备: {cfg.augmentation.device}")
    
    # 启用的策略
    enabled_strategies = []
    if cfg.augmentation.generation.strategies.plm.enabled:
        enabled_strategies.append('PLM')
    if cfg.augmentation.generation.strategies.coseq.enabled:
        enabled_strategies.append('CoSeq')
    logger.info(f"启用策略: {', '.join(enabled_strategies)}")
    logger.info(f"每种策略变体数: {cfg.augmentation.generation.variants_per_strategy}")
    logger.info(f"质量阈值: {cfg.augmentation.generation.quality_control.quality_threshold}")
    
    # 🔧 修复：在try块外初始化变量
    augmenter = None
    augmented_sequences = []
    
    try:
        # 加载数据
        products, sequences = load_data(cfg)
        monitor_memory()
        
        # 创建增强器
        logger.info("初始化增强器...")
        augmenter = RelationAugmenter(
            products=products,
            sequences=sequences,
            config=cfg
        )
        monitor_memory()
        
        # 生成增强序列
        logger.info("开始生成增强序列...")
        start_time = datetime.now()
        
        augmented_sequences = augmenter.generate_augmented_sequences()
        
        # 计算耗时
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"生成完成，耗时: {elapsed:.2f}秒")
        
        # 🔧 修复：在删除前保存需要的数据
        item2idx = getattr(augmenter, 'item2idx', {})
        filter_stats = getattr(augmenter, 'last_filter_stats', None)
        
        # 内存清理
        del augmenter
        augmenter = None  # 明确设置为None
        gc.collect()
        torch.cuda.empty_cache()
        
        # 统计信息
        logger.info("\n=== 生成统计 ===")
        logger.info(f"总序列数: {len(augmented_sequences)}")
        
        strategy_counts = defaultdict(lambda: {'count': 0, 'qualities': []})
        quality_scores = []
        
        for seq in augmented_sequences:
            strategy = seq['strategy']
            quality = seq['quality_score']
            
            strategy_counts[strategy]['count'] += 1
            strategy_counts[strategy]['qualities'].append(quality)
            quality_scores.append(quality)
        
        # 计算每个策略的统计信息
        logger.info("\n策略分布:")
        for strategy, stats in strategy_counts.items():
            count = stats['count']
            qualities = stats['qualities']
            if qualities:
                avg_quality = sum(qualities) / len(qualities)
                min_quality = min(qualities)
                max_quality = max(qualities)
                logger.info(f"  {strategy.upper()}:")
                logger.info(f"    - 数量: {count}")
                logger.info(f"    - 平均质量: {avg_quality:.3f}")
                logger.info(f"    - 质量范围: {min_quality:.3f} - {max_quality:.3f}")
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            logger.info(f"\n整体质量统计:")
            logger.info(f"  - 平均质量: {avg_quality:.3f}")
            logger.info(f"  - 质量范围: {min_quality:.3f} - {max_quality:.3f}")
        
        # 序列长度分布
        length_distribution = defaultdict(int)
        for seq in augmented_sequences:
            length = len(seq['augmented'])
            length_distribution[length] += 1
        
        logger.info(f"\n增强序列长度分布:")
        for length in sorted(length_distribution.keys())[:10]:
            logger.info(f"  长度 {length}: {length_distribution[length]} 个序列")
        
        # 保存结果
        output_dir = Path(cfg.data.augmented_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / cfg.data.fixed_aug_file
        
        # 计算策略统计
        strategy_stats = {}
        for strategy, stats in strategy_counts.items():
            qualities = stats['qualities']
            strategy_stats[strategy] = {
                'count': stats['count'],
                'avg_quality': sum(qualities) / len(qualities) if qualities else 0.0,
                'min_quality': min(qualities) if qualities else 0.0,
                'max_quality': max(qualities) if qualities else 0.0
            }
        
        save_data = {
            'pairs': augmented_sequences,
            'item2idx': item2idx,  # 🔧 修复：使用保存的item2idx
            'metadata': {
                'total_pairs': len(augmented_sequences),
                'strategy_counts': strategy_stats,
                'avg_quality': avg_quality if quality_scores else 0.0,
                'min_quality': min_quality if quality_scores else 0.0,
                'max_quality': max_quality if quality_scores else 0.0,
                'generation_time': elapsed,
                'target_count': cfg.target_count,
                'actual_count': len(augmented_sequences),
                'completion_rate': len(augmented_sequences) / cfg.target_count * 100,
                'version': 'v2.0_complete',
                'filter_applied': bool(filter_stats),
                'filter_stats': filter_stats,
                'config': {
                    'seed': seed,
                    'enabled_strategies': enabled_strategies,
                    'variants_per_strategy': cfg.augmentation.generation.variants_per_strategy,
                    'quality_threshold': cfg.augmentation.generation.quality_control.quality_threshold
                }
            }
        }
        
        torch.save(save_data, output_path)
        logger.info(f"✅ 结果保存至: {output_path}")
        
        # 最终统计
        completion_rate = len(augmented_sequences) / cfg.target_count * 100
        logger.info(f"\n=== 最终统计 ===")
        logger.info(f"目标数量: {cfg.target_count}")
        logger.info(f"实际生成: {len(augmented_sequences)}")
        logger.info(f"完成率: {completion_rate:.1f}%")
        logger.info(f"平均质量: {avg_quality:.3f}")
        logger.info(f"总耗时: {elapsed:.2f}秒")
        logger.info(f"生成速度: {len(augmented_sequences)/elapsed:.1f} 序列/秒")
        
        monitor_memory()
        
    except Exception as e:
        logger.error(f"增强失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
