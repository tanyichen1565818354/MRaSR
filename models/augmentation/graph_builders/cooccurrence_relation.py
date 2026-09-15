import os
from .base_relation import BaseRelationBuilder
import random
import numpy as np
from tqdm import tqdm

from models.augmentation.behavior_weight_utils import get_behavior_edge_multiplier


class CooccurrenceRelationBuilder(BaseRelationBuilder):
    def __init__(self, products, sequences, item2idx, config, behavior_relation_matrix=None):
        super().__init__(products, sequences, item2idx, config)
        self.behavior_relation_matrix = behavior_relation_matrix

    def build(self, **kwargs):
        config_params = self.config.relation_building.cooccurrence
        window_size = kwargs.get('window_size', config_params.get('window_size', 2))
        weight = kwargs.get('weight', config_params.get('default_weight', 0.6))
        sample_rate = kwargs.get('sample_rate', config_params.get('sample_rate', 0.3))

        seed = self.config.get("seed", 2025)
        random.seed(seed + os.getpid())
        np.random.seed(seed + os.getpid())

        # max_sequences 限制：只取前 N 条序列，避免全量遍历过慢
        max_sequences = self.config.get("graph_sampling", {}).get("max_sequences", None)
        all_seq_data = list(self.sequences.values())
        if max_sequences and len(all_seq_data) > max_sequences:
            rng = random.Random(seed)
            rng.shuffle(all_seq_data)
            all_seq_data = all_seq_data[:max_sequences]

        processed = 0
        total_seqs = len(all_seq_data)
        with tqdm(total=total_seqs, desc="[关系构建] 共现关系(排除特殊token)", bar_format="{l_bar}{bar:20}{r_bar}", leave=False) as pbar:
            for seq_data in all_seq_data:
                sequence = seq_data['sequence']
                # 多行为：同步提取 behaviors；缺失时默认 click(0)，与 learner 一致
                behaviors = seq_data.get('behaviors', [0] * len(sequence))

                filtered_sequence = []
                filtered_behaviors = []
                for item, beh in zip(sequence, behaviors):
                    if item == '<PAD>':
                        continue
                    if item in self.item2idx and self.item2idx[item] in self.special_token_indices:
                        continue
                    if item in self.item2idx:
                        filtered_sequence.append(item)
                        filtered_behaviors.append(beh)

                for i in range(len(filtered_sequence)):
                    for j in range(i+1, min(i+window_size+1, len(filtered_sequence))):
                        item1 = filtered_sequence[i]
                        item2 = filtered_sequence[j]
                        if (item1 != '<PAD>' and item2 != '<PAD>' and
                            self.item2idx.get(item1, -1) not in self.special_token_indices and
                            self.item2idx.get(item2, -1) not in self.special_token_indices):
                            # 行为加权：static=w_src×w_dst；learned=σ(R[src,dst])
                            beh_mult = get_behavior_edge_multiplier(
                                self.config,
                                filtered_behaviors[i],
                                filtered_behaviors[j],
                                self.behavior_relation_matrix,
                            )
                            self.add_relation(item1, item2, 'co_occurrence', weight * beh_mult)
                processed += 1
                if processed % 100 == 0:
                    pbar.update(100)
                    pbar.refresh()
            pbar.update(total_seqs % 100)
            pbar.close()
        return self.get_edges()