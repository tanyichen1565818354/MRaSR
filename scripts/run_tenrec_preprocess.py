"""
Tenrec 多行为数据集预处理脚本。

用法：
    python scripts/run_tenrec_preprocess.py
    python scripts/run_tenrec_preprocess.py dataset=qk_article
    python scripts/run_tenrec_preprocess.py dataset=qk_video
    python scripts/run_tenrec_preprocess.py dataset=sbr_1m
"""
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.data_utils.tenrec_preprocessor import TenrecPreprocessor


@hydra.main(version_base=None, config_path="../configs", config_name="preprocess")
def main(cfg: DictConfig):
    seed = cfg.get("seed", 2025)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    preprocessor = TenrecPreprocessor(cfg)
    sequences, products, item2idx = preprocessor.full_pipeline()

    if cfg.get("save_sample", False):
        sample_path = Path(cfg.data_split.train_dir) / "sample_sequences.pth"
        sample = dict(list(sequences.items())[:100])
        torch.save(sample, sample_path)
        print(f"样例数据已保存: {sample_path}")


if __name__ == "__main__":
    main()
