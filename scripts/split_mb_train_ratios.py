#!/usr/bin/env python3
"""Sample 30% / 70% user subsets from a full Tenrec training split.

The same shuffled user order is used for both ratios, so the 0.3 subset is
contained in the 0.7 subset. This matches the sparsity protocol in the paper.

Run from the repository root:

  python scripts/split_mb_train_ratios.py --train-preprocessed data/QB-video/train/preprocessed
  python scripts/split_mb_train_ratios.py --train-preprocessed data/QK-article/train/preprocessed --seed 42

Writes:

  {dataset_root}/train_0.3/preprocessed/
  {dataset_root}/train_0.7/preprocessed/
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def _save_split(
    sequences: dict,
    products: dict,
    item2idx: dict,
    out_dir: Path,
    user_ids: list,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sequences, out_dir / "sequences.pth")
    torch.save(products, out_dir / "products.pth")
    torch.save(item2idx, out_dir / "item2idx.pth")
    torch.save(user_ids, out_dir / "user_ids.pth")


def _sample_users(
    full_sequences: dict,
    ratio: float,
    seed: int,
) -> tuple[dict, list]:
    users = sorted(full_sequences.keys())
    rng = random.Random(seed)
    rng.shuffle(users)
    keep_n = max(1, int(round(len(users) * ratio)))
    kept = users[:keep_n]
    sub = {u: full_sequences[u] for u in kept}
    return sub, kept


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export train_0.3 / train_0.7 user subsets from a full Tenrec training split"
    )
    p.add_argument(
        "--train-preprocessed",
        type=Path,
        required=True,
        help="Full training preprocessed dir (sequences.pth, products.pth, item2idx.pth)",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root that will contain train_0.3/preprocessed. Default: two levels above --train-preprocessed",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Subsampling seed",
    )
    p.add_argument(
        "--ratios",
        type=float,
        nargs=2,
        default=(0.3, 0.7),
        metavar=("R0", "R1"),
        help="Two user-keep ratios, default 0.3 0.7",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.train_preprocessed.resolve()
    for name in ("sequences.pth", "products.pth", "item2idx.pth"):
        if not (src / name).exists():
            print(f"Missing file: {src / name}", file=sys.stderr)
            sys.exit(1)

    r0, r1 = args.ratios
    for r in (r0, r1):
        if not (0 < r <= 1):
            print(f"Ratio must be in (0, 1]: {r}", file=sys.stderr)
            sys.exit(1)

    if args.dataset_root is None:
        if src.name != "preprocessed":
            print(
                "--train-preprocessed should usually point to .../train/preprocessed; "
                "if the directory name is not preprocessed, pass --dataset-root.",
                file=sys.stderr,
            )
        dataset_root = src.parent.parent
    else:
        dataset_root = args.dataset_root.resolve()

    sequences: dict = torch.load(src / "sequences.pth", map_location="cpu", weights_only=False)
    products: dict = torch.load(src / "products.pth", map_location="cpu", weights_only=False)
    item2idx: dict = torch.load(src / "item2idx.pth", map_location="cpu", weights_only=False)

    n_full = len(sequences)
    print(f"Source: {src}")
    print(f"Full training users: {n_full}, seed={args.seed}")

    out0 = dataset_root / f"train_{r0}" / "preprocessed"
    out1 = dataset_root / f"train_{r1}" / "preprocessed"

    sub0, users0 = _sample_users(sequences, r0, args.seed)
    sub1, users1 = _sample_users(sequences, r1, args.seed)

    _save_split(sub0, products, item2idx, out0, users0)
    _save_split(sub1, products, item2idx, out1, users1)

    emb_src = src / "embeddings"
    if emb_src.is_dir():
        for out_dir in (out0, out1):
            shutil.copytree(emb_src, out_dir / "embeddings", dirs_exist_ok=True)
    rc = src / "relation_cache.pth"
    if rc.is_file():
        for out_dir in (out0, out1):
            shutil.copy2(rc, out_dir / "relation_cache.pth")

    print("Done")
    print(f"  train_{r0}/preprocessed: {len(sub0)} users -> {out0}")
    print(f"  train_{r1}/preprocessed: {len(sub1)} users -> {out1}")
    print("  Point data.train_dir (or data_split.train_dir) at one of these paths; keep val/test unchanged.")


if __name__ == "__main__":
    main()
