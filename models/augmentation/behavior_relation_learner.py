"""从训练序列中学习可学习行为转移矩阵 R。

学习完成后，图谱边权重使用：
    edge_weight = base_weight × σ(R[beh_src, beh_dst])

与 static 模式（w[src] × w[dst]）区分，R 建模的是行为对 (src→dst) 的联合强度。
"""
from __future__ import annotations

import logging
import random
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

BEHAVIOR_NAMES = {
    0: "click",
    1: "like",
    2: "comment",
    3: "follow",
    4: "share",
    5: "favorite",
    6: "read",
}


class BehaviorRelationLearner:
    """从多行为序列中学习 num_behaviors × num_behaviors 转移矩阵。"""

    def __init__(self, num_behaviors: int = 7, device: str = "cpu"):
        self.num_behaviors = int(num_behaviors)
        self.device = torch.device(device)

    def learn(self, sequences: Dict, config) -> torch.Tensor:
        mb_cfg = config.get("multi_behavior", {})
        learn_cfg = mb_cfg.get("learn", {})
        epochs = int(learn_cfg.get("epochs", 30))
        lr = float(learn_cfg.get("lr", 0.05))
        co_window = int(learn_cfg.get("cooccurrence_window", 2))
        co_weight = float(learn_cfg.get("cooccurrence_loss_weight", 0.3))
        max_sequences = config.get("graph_sampling", {}).get("max_sequences", None)
        seed = int(config.get("seed", 2025))

        seq_pairs = self._collect_behavior_pairs(
            sequences,
            co_window=co_window,
            max_sequences=max_sequences,
            seed=seed,
        )
        if not seq_pairs["sequential"]:
            logger.warning("未收集到行为转移样本，使用零矩阵初始化 R")
            return torch.zeros(self.num_behaviors, self.num_behaviors)

        init_logits = self._initialize_logits_from_counts(seq_pairs)
        relation_logits = nn.Parameter(init_logits.clone().to(self.device))
        optimizer = torch.optim.Adam([relation_logits], lr=lr)

        seq_src, seq_dst = self._pairs_to_tensors(seq_pairs["sequential"])
        co_src, co_dst = self._pairs_to_tensors(seq_pairs["cooccurrence"])

        logger.info(
            f"开始学习行为转移矩阵: sequential={len(seq_src)}, "
            f"cooccurrence={len(co_src)}, epochs={epochs}, lr={lr}"
        )

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=self.device)

            if len(seq_src) > 0:
                logits = relation_logits[seq_src]
                loss = loss + F.cross_entropy(logits, seq_dst)

            if len(co_src) > 0 and co_weight > 0:
                # 共现项：鼓励共现行为对在 R 上有更高联合强度
                co_scores = torch.sigmoid(relation_logits[co_src, co_dst])
                co_loss = -torch.log(co_scores.clamp(min=1e-6)).mean()
                loss = loss + co_weight * co_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
                logger.info(f"  [行为矩阵学习] epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")

        learned_matrix = relation_logits.detach().cpu()
        self._log_matrix_summary(learned_matrix)

        # Export R for paper Fig. 3 (heatmap). Enabled when learn.save_path is set in config.
        save_path = learn_cfg.get("save_path", None)
        if save_path:
            from pathlib import Path
            out = Path(save_path)
            if not out.is_absolute():
                # Prefer workspace root (parent of RDR4SR) so paper/scripts/... works under Hydra.
                repo_root = Path(__file__).resolve().parents[2]  # RDR4SR/
                workspace = repo_root.parent
                cand = workspace / out
                out = cand if str(out).startswith("paper/") else (repo_root / out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(learned_matrix, out)
            logger.info(f"  [行为矩阵学习] R 已导出 -> {out}")
        return learned_matrix

    def _collect_behavior_pairs(
        self,
        sequences: Dict,
        co_window: int,
        max_sequences: int | None,
        seed: int,
    ) -> Dict[str, Iterable[Tuple[int, int]]]:
        all_seq_data = list(sequences.values())
        if max_sequences and len(all_seq_data) > max_sequences:
            rng = random.Random(seed)
            rng.shuffle(all_seq_data)
            all_seq_data = all_seq_data[:max_sequences]

        sequential_pairs = []
        cooccurrence_pairs = []

        for seq_data in all_seq_data:
            sequence = seq_data.get("sequence", [])
            behaviors = seq_data.get("behaviors", [0] * len(sequence))
            if len(sequence) != len(behaviors):
                continue

            filtered_beh = []
            for item, beh in zip(sequence, behaviors):
                if item and str(item).strip() and str(item) != "<PAD>":
                    filtered_beh.append(int(beh))

            for i in range(len(filtered_beh) - 1):
                sequential_pairs.append((filtered_beh[i], filtered_beh[i + 1]))

            for i in range(len(filtered_beh)):
                for j in range(i + 1, min(i + co_window + 1, len(filtered_beh))):
                    if i != j:
                        cooccurrence_pairs.append((filtered_beh[i], filtered_beh[j]))

        return {
            "sequential": sequential_pairs,
            "cooccurrence": cooccurrence_pairs,
        }

    def _initialize_logits_from_counts(self, seq_pairs: Dict) -> torch.Tensor:
        counts = torch.ones(self.num_behaviors, self.num_behaviors)
        for src, dst in seq_pairs["sequential"]:
            if 0 <= src < self.num_behaviors and 0 <= dst < self.num_behaviors:
                counts[src, dst] += 1.0
        row_sum = counts.sum(dim=1, keepdim=True)
        probs = counts / row_sum.clamp(min=1.0)
        return torch.log(probs.clamp(min=1e-6))

    def _pairs_to_tensors(self, pairs: Iterable[Tuple[int, int]]):
        if not pairs:
            return (
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
            )
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=self.device)
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=self.device)
        return src, dst

    def _log_matrix_summary(self, relation_logits: torch.Tensor) -> None:
        relation_probs = torch.sigmoid(relation_logits)
        logger.info("✅ 行为转移矩阵学习完成（σ(R) 摘要）")
        header = "src\\dst " + " ".join(f"{b:>8}" for b in range(self.num_behaviors))
        logger.info(header)
        for i in range(self.num_behaviors):
            row = " ".join(f"{relation_probs[i, j].item():8.3f}" for j in range(self.num_behaviors))
            src_name = BEHAVIOR_NAMES.get(i, str(i))
            logger.info(f"{src_name:>7} {row}")
