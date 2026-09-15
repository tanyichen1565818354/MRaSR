# models/augmentation/graph_builders/sequential_relation.py
from .base_relation import BaseRelationBuilder
import random
import os
import numpy as np

from models.augmentation.behavior_weight_utils import get_behavior_edge_multiplier


class SequentialRelationBuilder(BaseRelationBuilder):
    def __init__(self, products, sequences, item2idx, config, behavior_relation_matrix=None):
        super().__init__(products, sequences, item2idx, config)
        self.behavior_relation_matrix = behavior_relation_matrix

    def build(self, **kwargs):
        weight = kwargs.get('weight', self.config.relation_building.sequential.default_weight)
        decay_factor = kwargs.get('decay_factor', self.config.relation_building.sequential.decay_factor)

        seed = self.config.get("seed", 2025)
        # max_sequences 限制
        max_sequences = self.config.get("graph_sampling", {}).get("max_sequences", None)
        all_seq_data = list(self.sequences.values())
        if max_sequences and len(all_seq_data) > max_sequences:
            rng = random.Random(seed)
            rng.shuffle(all_seq_data)
            all_seq_data = all_seq_data[:max_sequences]

        for seq_data in all_seq_data:
            sequence = seq_data['sequence']
            # 多行为：同步提取 behaviors 列表，不存在时全部视为 purchase(2)
            behaviors = seq_data.get('behaviors', [2] * len(sequence))

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

            for i in range(len(filtered_sequence) - 1):
                current_item = filtered_sequence[i]
                next_item = filtered_sequence[i + 1]
                if (current_item != '<PAD>' and next_item != '<PAD>' and
                    self.item2idx.get(current_item, -1) not in self.special_token_indices and
                    self.item2idx.get(next_item, -1) not in self.special_token_indices):
                    time_weight = weight * (decay_factor ** (i + 1))
                    # 行为加权：static=w_src×w_dst；learned=σ(R[src,dst])
                    beh_mult = get_behavior_edge_multiplier(
                        self.config,
                        filtered_behaviors[i],
                        filtered_behaviors[i + 1],
                        self.behavior_relation_matrix,
                    )
                    self.add_relation(current_item, next_item, 'sequential', time_weight * beh_mult)
        return self.get_edges()
