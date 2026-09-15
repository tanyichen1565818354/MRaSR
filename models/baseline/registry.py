"""多行为序列推荐 baseline 模型注册表。

run_train.py 与 convert_embeddings.py 共用此注册表，避免重复维护。
当前注册的多行为 baseline：
  - SASBAR : SASRec 的多行为扩展（行为嵌入 + 目标行为条件化查询）
  - GRUBAR : GRU4Rec 的多行为扩展
  - MBHT   : 多行为异构 Transformer
  - RLBL   : Recurrent Log-BiLinear model（行为特定转移矩阵 + RNN 聚合）
  - RIB    : Recommendation from the micro-behavior perspective（GRU + 行为注意力）
  - BINN   : Behavior-Intensive Neural Network（CLSTM + Bi-CLSTM 双路长短期偏好）
"""
import importlib


_MODEL_REGISTRY = {
    "sasbar": ("models.baseline.sasbar.SASBAR", "SASBAR"),
    "grubar": ("models.baseline.grubar.GRUBAR", "GRUBAR"),
    "mbht":   ("models.baseline.mbht.MBHT",   "MBHT"),
    "rlbl":   ("models.baseline.rlbl.RLBL",   "RLBL"),
    "rib":    ("models.baseline.rib.RIB",     "RIB"),
    "binn":   ("models.baseline.binn.BINN",   "BINN"),
}


def get_model_class(model_name: str):
    """根据模型名返回模型类。"""
    key = model_name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    module_path, class_name = _MODEL_REGISTRY[key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def available_models():
    return sorted(_MODEL_REGISTRY.keys())
