"""多行为图谱构建：行为权重工具。

支持两种模式（由 config.multi_behavior.mode 控制）：
  - static         : 手调标量 behavior_weights，边权重 = base × w[src] × w[dst]
  - learned_matrix : 可学习行为转移矩阵 R，边权重 = base × σ(R[src, dst])
"""
from __future__ import annotations

from typing import Optional

import torch


def get_multi_behavior_mode(config) -> str:
    try:
        mb_cfg = config.get("multi_behavior", None)
        if mb_cfg and mb_cfg.get("enabled", False):
            return str(mb_cfg.get("mode", "static"))
    except Exception:
        pass
    return "static"


def get_behavior_edge_multiplier(
    config,
    beh_src: int,
    beh_dst: int,
    relation_matrix: Optional[torch.Tensor] = None,
) -> float:
    """计算一条边上行为相关的权重乘子。"""
    try:
        mb_cfg = config.get("multi_behavior", None)
        if not mb_cfg or not mb_cfg.get("enabled", False):
            return 1.0

        mode = str(mb_cfg.get("mode", "static"))
        if mode == "learned_matrix" and relation_matrix is not None:
            num_beh = relation_matrix.shape[0]
            if 0 <= beh_src < num_beh and 0 <= beh_dst < num_beh:
                return float(torch.sigmoid(relation_matrix[beh_src, beh_dst]).item())
            return 1.0

        # static：标量权重乘积（旧方案，用于消融）
        bw = mb_cfg.get("behavior_weights", {})
        w_src = float(bw.get(str(beh_src), 1.0))
        w_dst = float(bw.get(str(beh_dst), 1.0))
        return w_src * w_dst
    except Exception:
        return 1.0
