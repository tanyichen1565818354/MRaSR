# scripts/generate_plm_emb.py
import os
import re
import tempfile
import torch
from html import unescape
import hydra
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import logging
import random
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def _torch_save_atomic(obj, path: str) -> None:
    """Write via a temp file then replace, avoiding torn files on full disk / NFS."""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".pth.tmp", dir=parent)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def clean_text(text: str) -> str:
    """增强型文本清洗函数"""
    # 处理HTML转义字符
    cleaned = unescape(text)
    # 移除特殊符号但保留关键语义符号
    cleaned = re.sub(r'[^\w\s\-\'",.;:!?()]', ' ', cleaned)
    # 合并连续空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def _discover_emb_targets(config):
    """枚举所有需要写入 PLM 嵌入的目录。

    Tenrec preprocessing writes ``data/<dataset>/train|val|test/preprocessed``.
    Sparse-protocol splits ``train_0.3`` / ``train_0.7`` are created later by
    ``scripts/split_mb_train_ratios.py``. Baseline configs default to
    ``train/preprocessed``. Every existing ``<split>/preprocessed`` directory is
    treated as a write target so item2idx and PLM embeddings stay aligned.

    Override with ``PLM_EMBEDDINGS_DIR=/abs/path`` to write a single directory.
    """
    ds = config.data_split
    train_dir = os.path.abspath(ds.train_dir)

    emb_override = (os.environ.get("PLM_EMBEDDINGS_DIR") or "").strip()
    if emb_override:
        emb_targets = [("override", os.path.abspath(os.path.expanduser(emb_override)))]
        logger.info(
            f"使用 PLM_EMBEDDINGS_DIR={emb_targets[0][1]}（请保证图构建配置 plm_emb_path 指向相同路径）"
        )
        return emb_targets

    # train_dir 形如 .../data/<dataset>/train/preprocessed；dataset 根 = 上两级
    split_root = os.path.dirname(os.path.dirname(train_dir))
    emb_targets = []

    if os.path.isdir(split_root):
        for name in sorted(os.listdir(split_root)):
            split_dir = os.path.join(split_root, name, "preprocessed")
            if os.path.isdir(split_dir):
                emb_targets.append((name, os.path.abspath(os.path.join(split_dir, "embeddings"))))
    # 兜底：如果目录扫描没命中（旧结构），至少保留 train/val/test
    if not emb_targets:
        emb_targets.append(("train", os.path.abspath(os.path.join(train_dir, "embeddings"))))
        for tag, key in (("val", "val_dir"), ("test", "test_dir")):
            d = ds.get(key)
            if d:
                emb_targets.append((tag, os.path.abspath(os.path.join(d, "embeddings"))))

    logger.info(
        "PLM 嵌入将写入: " + ", ".join(f"{tag}={d}" for tag, d in emb_targets)
    )
    return emb_targets


_BEHAVIOR_NAMES = {
    0: "click", 1: "like", 2: "comment", 3: "follow",
    4: "share", 5: "favorite", 6: "read",
}


def _build_item_text(asin: str, meta: dict) -> str:
    """根据商品元数据构建 PLM 输入文本。

    Tenrec 的 products.pth 没有自然语言元数据，但预处理时聚合了 per-item
    统计画像（behavior_counts / avg_watching_times / avg_gender / avg_age /
    total_interactions）。这里把统计画像转成自然语言句子喂给 PLM，
    使得"受众和参与模式相似的 item"获得相近的 768 维向量。

    If a dataset still has natural-language title/brand/description fields,
    those are preferred over the Tenrec statistical profile.
    """
    # ── 旧数据集兼容：有真实 title 时走富文本路径 ──────────────────
    title = str(meta.get('title', '') or '').strip()
    if title and not title.startswith('Item '):
        parts = [clean_text(title)]
        brand = str(meta.get('brand', '') or '').strip()
        if brand and brand.lower() != 'unknown':
            parts.append(clean_text(brand))
        desc = str(meta.get('description', '') or '').strip()
        if desc and not desc.startswith('Categories '):
            parts.append(clean_text(desc))
        cats = meta.get('categories') or []
        if cats and isinstance(cats[0], list):
            parts.append(clean_text(' '.join(str(c) for c in cats[0])))
        return ' | '.join(p for p in parts if p)

    # ── Tenrec 路径：统计画像 → 自然语言 ──────────────────────────
    # category
    cats = meta.get('categories') or []
    if cats and isinstance(cats[0], list):
        cat_val = cats[0][0] if cats[0] else "unknown"
    elif cats:
        cat_val = cats[0]
    else:
        cat_val = "unknown"

    # 行为分布
    behavior_counts = meta.get('behavior_counts', {})
    total = meta.get('total_interactions', 0)

    if total > 0 and behavior_counts:
        # 按行为频次降序列出
        beh_parts = []
        for beh_id in sorted(behavior_counts.keys()):
            name = _BEHAVIOR_NAMES.get(beh_id, f"behavior_{beh_id}")
            cnt = behavior_counts[beh_id]
            pct = cnt / total * 100
            beh_parts.append(f"{pct:.0f}% {name} ({cnt} times)")

        beh_text = ", ".join(beh_parts)
    else:
        beh_text = "no interaction data"

    # 观看时长
    avg_wt = meta.get('avg_watching_times', 0.0)
    if avg_wt > 0:
        wt_text = f"Average watching time is {avg_wt:.0f} units."
    else:
        wt_text = ""

    # 用户画像
    avg_g = meta.get('avg_gender', -1)
    avg_a = meta.get('avg_age', -1)
    demo_parts = []
    if avg_g >= 0:
        gender_label = "male" if avg_g > 0.5 else ("female" if avg_g < 0.5 else "mixed gender")
        demo_parts.append(f"Most viewers are {gender_label} (gender score {avg_g:.2f})")
    if avg_a >= 0:
        demo_parts.append(f"in age group {avg_a:.0f}")
    demo_text = " ".join(demo_parts) + "." if demo_parts else ""

    # 拼成完整句子
    sentences = [f"Video item {asin} in category {cat_val}."]
    sentences.append(f"It receives {total} total interactions: {beh_text}.")
    if wt_text:
        sentences.append(wt_text)
    if demo_text:
        sentences.append(demo_text)

    return " ".join(sentences)


def generate_plm_embeddings(config, device="cuda:0"):
    """
    改进版PLM嵌入生成流程，支持多数据集（Tenrec 版本以 category 为主输入）
    """
    # 设置随机种子
    seed = config.get("seed", 2025)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        # === 识别数据目录 ===
        ds = config.data_split
        train_dir = ds.train_dir
        emb_targets = _discover_emb_targets(config)
        
        # === 数据加载 ===
        # 从训练集加载商品数据和item2idx映射
        products_path = os.path.join(train_dir, "products.pth")
        item2idx_path = os.path.join(train_dir, "item2idx.pth")
        
        if not os.path.exists(products_path):
            raise FileNotFoundError(f"商品数据文件缺失: {products_path}")
        if not os.path.exists(item2idx_path):
            raise FileNotFoundError(f"item2idx文件缺失: {item2idx_path}")
            
        products = torch.load(products_path, weights_only=False)
        item2idx = torch.load(item2idx_path, weights_only=False)
        logger.info(f"成功加载 {len(products)} 个商品元数据")
        logger.info(f"成功加载 {len(item2idx)} 个物品索引映射")
        
        # === 过滤实际物品 ===
        # 只为实际物品生成嵌入，排除PAD
        actual_items = [asin for asin in item2idx.keys() if asin != '<PAD>']
        logger.info(f"实际物品数量: {len(actual_items)} (排除PAD)")
        
        # Build PLM text: Tenrec statistical profile, or title/brand if present.
        texts = []
        product_ids = []
        missing_text = 0

        for asin in actual_items:
            if asin not in products:
                logger.warning(f"商品 {asin} 在products中不存在，跳过")
                continue

            meta = products[asin]
            product_ids.append(asin)
            item_text = _build_item_text(asin, meta)

            if not item_text.strip():
                missing_text += 1
                item_text = f"item {asin}"

            texts.append(item_text)

        logger.info(f"缺失有效文本的样本数: {missing_text} ({missing_text/len(product_ids):.1%})")
        
        # === 文本验证 ===
        sample_texts = texts[:5] + texts[-5:]
        logger.info("文本样例检查：")
        for i, t in enumerate(sample_texts):
            logger.info(f"Sample {i+1}: {t[:100]}...")

        # === 模型初始化 ===
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_name = "models/all-mpnet-base-v2"
        logger.info(f"正在加载模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        
        # === 批量编码 ===
        batch_size = 64
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="生成PLM嵌入"):
                batch_texts = texts[i:i+batch_size]
                
                # 编码文本
                inputs = tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(device)
                
                outputs = model(**inputs)
                # 使用[CLS] token的嵌入
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                embeddings.append(batch_embeddings)
        
        # 合并所有嵌入
        emb_tensor = torch.cat(embeddings, dim=0)
        logger.info(f"嵌入矩阵维度: {emb_tensor.shape}")
        
        # === 保存结果（与 graph 一致：item_embeddings + item_id2idx）===
        # 切勿对 20 万+ 物品再 torch.save 巨型 {id: tensor}：体积/内存暴涨，易触发 disk full 与
        # PytorchStreamWriter / unexpected pos 写入错误。需要按 id 取向量时用 emb_tensor[id2idx[id]]。
        id2idx = {pid: idx for idx, pid in enumerate(product_ids)}
        for tag, emb_dir in emb_targets:
            _torch_save_atomic(emb_tensor, os.path.join(emb_dir, "item_embeddings.pth"))
            _torch_save_atomic(id2idx, os.path.join(emb_dir, "item_id2idx.pth"))
            logger.info(f"[{tag}] PLM嵌入已保存至 {emb_dir}")
        
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        raise

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../configs", config_name="preprocess")
    def main(cfg: DictConfig):
        device = cfg.get("device", "cuda:0")
        generate_plm_embeddings(cfg, device=device)

    main()