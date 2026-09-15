from pathlib import Path
import torch
import logging
from tqdm import tqdm
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from models.baseline.registry import get_model_class

import random
import numpy as np
import os

import hydra
from omegaconf import DictConfig

def generate_embeddings(model_name: str, dataset_name: str, model_path: str,
                        item_map_path: str, output_path: str, cuda_id: int = 0):
    """从训练好的多行为 baseline 检查点导出融合后的 item embedding。

    参数：
      model_name    : 模型名（sasbar / grubar / mbht）
      dataset_name  : 数据集名（仅用于日志）
      model_path    : best_model.ckpt 路径
      item_map_path : item2idx.pth 路径
      output_path   : 导出 embedding 的保存路径
    """
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tqdm(total=3, desc=f"生成 {model_name} 的 item embedding ({dataset_name})") as pbar:
        item2idx = torch.load(item_map_path, weights_only=False)
        pbar.update(1)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型检查点文件未找到: {model_path}")
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'parameters' not in ckpt or 'config' not in ckpt:
            raise ValueError("检查点文件格式不正确，需包含 'parameters' 和 'config' 键。")
        model_state_dict = ckpt['parameters']
        saved_config = ckpt['config']
        pbar.update(1)

        ModelClass = get_model_class(saved_config.model.name)
        model = ModelClass(saved_config).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        pbar.update(1)

        with torch.no_grad():
            if hasattr(model, 'get_item_embedding'):
                embeddings = model.get_item_embedding()
            else:
                embeddings = model.item_embedding.weight.clone().cpu()

    torch.save(embeddings, output_path)
    print(f"成功! 嵌入已保存至: {output_path}  shape={tuple(embeddings.shape)}")
    return embeddings


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../configs", config_name="convert_embeddings")
    def hydra_main(cfg: DictConfig):
        """从训练好的 baseline 检查点导出 item embedding。

        用法:
          python scripts/convert_embeddings.py model_name=grubar dataset_name=QB-video
          python scripts/convert_embeddings.py paths.model_path=自定义路径
        """
        model_name = cfg.model_name
        dataset_name = cfg.dataset_name
        cuda_id = cfg.cuda_id
        train_subdir = cfg.train_subdir

        def resolve_path(explicit, template_key):
            """优先用显式指定的路径，否则用模板自动推导。"""
            if explicit is not None:
                return str(explicit)
            template = cfg.path_templates[template_key]
            return template.format(
                model_upper=model_name.upper(),
                model_lower=model_name,
                dataset=dataset_name,
                train_subdir=train_subdir,
            )

        model_path = resolve_path(cfg.paths.model_path, 'model_path')
        item_map_path = resolve_path(cfg.paths.item_map_path, 'item_map_path')
        output_path = resolve_path(cfg.paths.output_path, 'output_path')

        print(f"模型: {model_name}, 数据集: {dataset_name}")
        print(f"  checkpoint: {model_path}")
        print(f"  item2idx:   {item_map_path}")
        print(f"  输出路径:    {output_path}")

        generate_embeddings(model_name, dataset_name, model_path,
                            item_map_path, output_path, cuda_id)

    hydra_main()
