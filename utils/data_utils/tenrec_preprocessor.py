"""
Tenrec 多行为数据集预处理器。

支持四个 Tenrec 数据集（schema 各异，通过 config 切换）：
  - sbr_data_1M : click, follow, like, share
  - QB-video    : click, follow, like, share
  - QK-video    : click, follow, like, share
  - QK-article  : click, read, share, like, follow, favorite

统一行为 ID 编码：
    {click:0, like:1, comment:2, follow:3, share:4, favorite:5, read:6}

一行可能有多个行为位为 1，按"最强行为优先"取单一行为：
    favorite > like > share > comment > follow > read > click

评估目标行为统一为 like (id=1)。留一法切分时按 like 位置切分，
测试期/验证期没有 like 目标的用户从对应 split 剔除。

输出 .pth 文件：
    sequences.pth / products.pth / item2idx.pth / user_ids.pth
每个用户序列的 dict 含：
    sequence        : List[str]   item_id 列表（按 CSV 行序）
    behaviors       : List[int]   与 sequence 对齐的行为 ID 列表
    features        : List[dict]  每步的元数据（category 等，供 PLM 替代用）
    total_items     : int
test/val/train split 还额外含 target_item / target_behavior。

注：test/val/train 的 sequence 字段**包含目标项**（即 input+target），
这样下游 create_dataloader(is_test=True) 用 seq[:-1] 作输入、seq[-1] 作目标
正好是 like 目标，与原管线一致且无 off-by-one。
"""

import csv
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# 统一行为 ID
BEHAVIOR_CLICK = 0
BEHAVIOR_LIKE = 1
BEHAVIOR_COMMENT = 2
BEHAVIOR_FOLLOW = 3
BEHAVIOR_SHARE = 4
BEHAVIOR_FAVORITE = 5
BEHAVIOR_READ = 6

BEHAVIOR_NAMES = {
    BEHAVIOR_CLICK: "click",
    BEHAVIOR_LIKE: "like",
    BEHAVIOR_COMMENT: "comment",
    BEHAVIOR_FOLLOW: "follow",
    BEHAVIOR_SHARE: "share",
    BEHAVIOR_FAVORITE: "favorite",
    BEHAVIOR_READ: "read",
}

# 最强行为优先（从强到弱）
BEHAVIOR_PRIORITY = [
    BEHAVIOR_FAVORITE,
    BEHAVIOR_LIKE,
    BEHAVIOR_SHARE,
    BEHAVIOR_COMMENT,
    BEHAVIOR_FOLLOW,
    BEHAVIOR_READ,
    BEHAVIOR_CLICK,
]

# 默认行为权重（用于图谱加权和增强保护，可被 config 覆盖）
DEFAULT_BEHAVIOR_WEIGHTS = {
    "0": 0.3,   # click
    "1": 1.0,   # like  (目标行为，权重最高)
    "2": 0.5,   # comment
    "3": 0.5,   # follow
    "4": 0.6,   # share
    "5": 0.8,   # favorite
    "6": 0.2,   # read
}

# 默认行为替换保护概率（值越小越不容易被增强替换）
DEFAULT_BEHAVIOR_REPLACE_PROBS = {
    "0": 0.20,  # click
    "1": 0.03,  # like  (目标行为，强保护)
    "2": 0.10,  # comment
    "3": 0.10,  # follow
    "4": 0.08,  # share
    "5": 0.05,  # favorite
    "6": 0.20,  # read
}


def _truthy(value) -> bool:
    """判断一个 CSV 字段是否表示"行为发生"。

    支持：
      - 数值 1 / 1.0 / >0
      - 字符串 'True' / 'true' / '1' / '1.0'
    """
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) > 0
    s = str(value).strip().lower()
    if s in ("true", "1", "1.0"):
        return True
    try:
        return float(s) > 0
    except ValueError:
        return False


class TenrecPreprocessor:
    """Tenrec 多行为序列预处理器（config 驱动，支持四个数据集）。"""

    def __init__(self, config: DictConfig):
        self.config = config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )

        # 数据集行为列定义：col_name -> behavior_id
        # 由 config.dataset.behavior_columns 提供
        self.behavior_columns: Dict[str, int] = dict(config.dataset.behavior_columns)
        # 该数据集涉及的行为 id 集合
        self.available_behaviors = set(self.behavior_columns.values())

        # 目标行为（评估用），默认 like(1)
        self.target_behavior: int = int(config.dataset.get("target_behavior", BEHAVIOR_LIKE))
        # 该数据集是否真的含目标行为
        if self.target_behavior not in self.available_behaviors:
            raise ValueError(
                f"目标行为 {self.target_behavior}({BEHAVIOR_NAMES[self.target_behavior]}) "
                f"不在数据集 {config.dataset.name} 的行为列中: {self.available_behaviors}"
            )

        # item 元数据列（用于 category 特征，替代 PLM 文本 embedding）
        self.category_columns: List[str] = list(
            config.dataset.get("category_columns", [])
        )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def full_pipeline(self):
        """完整数据处理流水线"""
        # 1. 加载 CSV，按用户聚合（保持行序）
        user_data = self._load_raw(self.config.raw.data_path)

        # 2. 构建多行为序列
        sequences = self._build_sequences(user_data)

        # 3. k-core 迭代过滤
        sequences = self._apply_kcore_filter(sequences)

        # 4. 商品元数据 + 索引映射
        products = self._build_products(sequences, user_data)
        item2idx = self._build_item2idx(products)

        # 5. 按 like 目标做留一法切分
        train_seqs, val_seqs, test_seqs = self._temporal_split_by_target(sequences)

        # 6. 保存
        self._save_split(train_seqs, products, item2idx, self.config.data_split.train_dir, "train")
        self._save_split(val_seqs, products, item2idx, self.config.data_split.val_dir, "val")
        self._save_split(test_seqs, products, item2idx, self.config.data_split.test_dir, "test")

        # 6.5 稀疏训练子集（0.7 / 0.3）
        try:
            base = self.config.data_split.train_dir
            parent = os.path.dirname(os.path.dirname(base))
            self._sample_users_and_save(
                train_seqs, products, item2idx,
                os.path.join(parent, "train_0.7", "preprocessed"), 0.7,
            )
            self._sample_users_and_save(
                train_seqs, products, item2idx,
                os.path.join(parent, "train_0.3", "preprocessed"), 0.3,
            )
        except Exception as e:
            logger.warning(f"生成稀疏训练子集失败: {e}")

        self._print_dataset_stats(sequences, products, train_seqs, val_seqs, test_seqs)
        return sequences, products, item2idx

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_raw(self, data_path: str) -> Dict[str, List[dict]]:
        """加载 Tenrec CSV，按 user_id 聚合，保持 CSV 行序。"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        user_data: Dict[str, List[dict]] = defaultdict(list)
        total = 0
        skipped = 0

        logger.info(f"加载数据: {data_path}")
        with open(data_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="读取 CSV"):
                try:
                    user_id = str(row["user_id"]).strip()
                    item_id = str(row["item_id"]).strip()
                    if not user_id or not item_id:
                        skipped += 1
                        continue

                    # 按优先级取最强行为
                    behavior = self._pick_strongest_behavior(row)
                    if behavior is None:
                        # 该行没有任何正向行为，跳过
                        skipped += 1
                        continue

                    # 商品元数据（category 字段）
                    categories = []
                    for col in self.category_columns:
                        val = row.get(col)
                        if val is not None and str(val).strip():
                            categories.append(str(val).strip())

                    # 辅助数值列（用于 per-item 统计画像，供 PLM 替代用）
                    watching_times = self._parse_float(row.get("watching_times"))
                    gender = self._parse_int(row.get("gender"))
                    age = self._parse_int(row.get("age"))

                    user_data[user_id].append({
                        "item_id": item_id,
                        "behavior": behavior,
                        "categories": categories,
                        "watching_times": watching_times,
                        "gender": gender,
                        "age": age,
                    })
                    total += 1
                except Exception:
                    skipped += 1

        logger.info(
            f"读取完成: {total} 条有效记录, {skipped} 条跳过, {len(user_data)} 个用户"
        )
        return user_data

    def _pick_strongest_behavior(self, row: dict) -> int:
        """按 BEHAVIOR_PRIORITY 顺序，返回该行最强的发生行为；无则返回 None。"""
        for beh in BEHAVIOR_PRIORITY:
            if beh not in self.available_behaviors:
                continue
            # 找到对应列名
            for col_name, beh_id in self.behavior_columns.items():
                if beh_id == beh and _truthy(row.get(col_name)):
                    return beh
        return None

    @staticmethod
    def _parse_float(val) -> float:
        """安全解析浮点数，失败返回 0.0。"""
        if val is None:
            return 0.0
        try:
            return float(str(val).strip())
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _parse_int(val) -> int:
        """安全解析整数，失败返回 -1（表示缺失）。"""
        if val is None:
            return -1
        try:
            return int(float(str(val).strip()))
        except (ValueError, TypeError):
            return -1

    def _build_sequences(self, user_data: Dict[str, List[dict]]) -> Dict[str, dict]:
        """按用户聚合为序列（保持行序），同时记录 behaviors 和 features。"""
        min_length = self.config.sequence.get("min_length", 3)
        max_length = self.config.sequence.get("max_length", 50)

        sequences = {}
        for user_id, records in tqdm(user_data.items(), desc="构建序列"):
            # CSV 行序即时间序（无时间戳，按出现顺序）
            items = [r["item_id"] for r in records]
            behaviors = [r["behavior"] for r in records]
            features = [
                {"asin": r["item_id"], "categories": r["categories"]}
                for r in records
            ]

            if len(items) < min_length:
                continue

            # 截断到 max_length（保留最近的）
            if len(items) > max_length:
                items = items[-max_length:]
                behaviors = behaviors[-max_length:]
                features = features[-max_length:]

            sequences[user_id] = {
                "sequence": items,
                "behaviors": behaviors,
                "features": features,
                "total_items": len(items),
            }

        logger.info(f"构建序列完成: {len(sequences)} 个有效用户")
        return sequences

    def _apply_kcore_filter(self, sequences: Dict[str, dict]) -> Dict[str, dict]:
        """k-core 迭代过滤（用户和商品都至少出现 k 次）。"""
        k = self.config.filtering.get("kcore", 5)
        logger.info(f"开始 {k}-core 过滤...")
        prev_len = -1
        iteration = 0

        while True:
            iteration += 1
            item_count = defaultdict(int)
            for seq_data in sequences.values():
                for item in seq_data["sequence"]:
                    item_count[item] += 1

            new_sequences = {}
            for user_id, seq_data in sequences.items():
                filtered = [
                    (item, beh, feat)
                    for item, beh, feat in zip(
                        seq_data["sequence"], seq_data["behaviors"], seq_data["features"]
                    )
                    if item_count[item] >= k
                ]
                if len(filtered) >= self.config.sequence.get("min_length", 3):
                    items, behaviors, features = zip(*filtered)
                    new_sequences[user_id] = {
                        "sequence": list(items),
                        "behaviors": list(behaviors),
                        "features": list(features),
                        "total_items": len(items),
                    }

            cur_len = len(new_sequences)
            logger.info(f"  第 {iteration} 轮: {cur_len} 个用户")
            if cur_len == prev_len:
                break
            prev_len = cur_len
            sequences = new_sequences

        logger.info(f"{k}-core 过滤完成: {len(sequences)} 个用户保留")
        return sequences

    def _build_products(
        self, sequences: Dict[str, dict], user_data: Dict[str, List[dict]]
    ) -> Dict[str, dict]:
        """构建商品元数据字典（含 per-item 统计画像，供 PLM 替代用）。"""
        # 从 user_data 收集每个 item 出现过的 categories 及数值特征
        item_categories: Dict[str, List[List[str]]] = defaultdict(list)
        item_stats: Dict[str, dict] = defaultdict(lambda: {
            "behavior_counts": defaultdict(int),
            "watching_times_sum": 0.0,
            "watching_times_cnt": 0,
            "gender_sum": 0,
            "gender_cnt": 0,
            "age_sum": 0,
            "age_cnt": 0,
        })
        for records in user_data.values():
            for r in records:
                iid = r["item_id"]
                if r["categories"]:
                    item_categories[iid].append(r["categories"])
                stats = item_stats[iid]
                stats["behavior_counts"][r["behavior"]] += 1
                wt = r.get("watching_times")
                if wt is not None and wt > 0:
                    stats["watching_times_sum"] += wt
                    stats["watching_times_cnt"] += 1
                g = r.get("gender")
                if g is not None and g >= 0:
                    stats["gender_sum"] += g
                    stats["gender_cnt"] += 1
                a = r.get("age")
                if a is not None and a >= 0:
                    stats["age_sum"] += a
                    stats["age_cnt"] += 1

        valid_items = set()
        for seq_data in sequences.values():
            valid_items.update(seq_data["sequence"])

        products = {}
        for item_id in valid_items:
            cats = item_categories.get(item_id, [])
            primary_cat = cats[0] if cats else []
            stats = item_stats.get(item_id)

            # 聚合统计画像
            behavior_counts = {}
            avg_watching = 0.0
            avg_gender = -1
            avg_age = -1
            total_interactions = 0
            if stats:
                behavior_counts = dict(stats["behavior_counts"])
                total_interactions = sum(behavior_counts.values())
                avg_watching = (
                    stats["watching_times_sum"] / stats["watching_times_cnt"]
                    if stats["watching_times_cnt"] > 0 else 0.0
                )
                avg_gender = (
                    stats["gender_sum"] / stats["gender_cnt"]
                    if stats["gender_cnt"] > 0 else -1
                )
                avg_age = (
                    stats["age_sum"] / stats["age_cnt"]
                    if stats["age_cnt"] > 0 else -1
                )

            products[item_id] = {
                "categories": [primary_cat] if primary_cat else [["unknown"]],
                "brand": "unknown",
                "title": f"Item {item_id}",
                "description": f"Categories {primary_cat}",
                "price": 0.0,
                # per-item 统计画像（供 generate_plm_emb.py 构造自然语言）
                "behavior_counts": behavior_counts,
                "avg_watching_times": round(avg_watching, 1),
                "avg_gender": round(avg_gender, 2),
                "avg_age": round(avg_age, 2),
                "total_interactions": total_interactions,
            }

        logger.info(f"商品元数据构建完成: {len(products)} 个商品（含统计画像）")
        return products

    def _build_item2idx(self, products: Dict[str, dict]) -> Dict[str, int]:
        """item -> idx 映射，PAD=0，商品从 1 开始。"""
        item2idx = {"<PAD>": 0}
        for idx, item_id in enumerate(sorted(products.keys(), key=lambda x: int(x) if x.isdigit() else x), start=1):
            item2idx[item_id] = idx
        logger.info(f"item2idx 构建完成: {len(item2idx)} 个 token (PAD + {len(products)} 商品)")
        return item2idx

    def _temporal_split_by_target(
        self, sequences: Dict[str, dict]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        多行为留一法切分（目标行为 = like）：

        训练集：所有用户都进入训练，next-item 预测在「留出目标之前的序列」上做，
                不限定目标行为。无 like 的用户也进训练（用完整序列），
                由 config.dataset.include_no_target_users_in_train 控制（默认 True）。
        验证集：≥2 个 like 的用户，留出「倒数第 2 个 like」作为目标。
        测试集：≥1 个 like 的用户，留出「最后 1 个 like」作为目标。

        val/test 的 sequence 字段包含目标项（input+target），
        下游 create_dataloader(is_test=True) 用 seq[:-1]/seq[-1] 正好命中 like 目标。
        train 的 sequence 字段是「留出目标之前」的序列（不含任何留出目标），
        下游 create_dataloader(is_test=False) 在其上枚举 next-item 样本。

        三个集合大小不同是多行为评估的正常现象：
        train ≥ test ≥ val（likes 稀疏导致 val/test 比 train 小）。
        """
        train_seqs, val_seqs, test_seqs = {}, {}, {}
        no_target_users = 0  # 无 like 的用户数

        include_no_target = bool(
            self.config.dataset.get("include_no_target_users_in_train", True)
        )
        min_train_len = self.config.sequence.get("min_train_length", 2)

        for user_id, data in sequences.items():
            seq = data["sequence"]
            beh = data["behaviors"]
            feat = data["features"]

            target_positions = [
                i for i, b in enumerate(beh) if b == self.target_behavior
            ]

            if len(target_positions) == 0:
                # 无目标行为：只能进训练集，用完整序列
                no_target_users += 1
                if include_no_target and len(seq) >= min_train_len:
                    train_seqs[user_id] = {
                        "sequence": seq,
                        "behaviors": beh,
                        "features": feat,
                        "total_items": len(seq),
                    }
                continue

            # test: 最后一个目标位置
            p_test = target_positions[-1]
            test_seqs[user_id] = {
                "sequence": seq[: p_test + 1],
                "behaviors": beh[: p_test + 1],
                "features": feat[: p_test + 1],
                "target_item": seq[p_test],
                "target_behavior": beh[p_test],
                "total_items": p_test + 1,
            }

            # val: 倒数第二个目标位置
            if len(target_positions) >= 2:
                p_val = target_positions[-2]
                val_seqs[user_id] = {
                    "sequence": seq[: p_val + 1],
                    "behaviors": beh[: p_val + 1],
                    "features": feat[: p_val + 1],
                    "target_item": seq[p_val],
                    "target_behavior": beh[p_val],
                    "total_items": p_val + 1,
                }
                train_end = p_val  # 训练用 val 目标之前的序列（不含 val/test 目标）
            else:
                train_end = p_test  # 无 val，训练用 test 目标之前的序列

            # train: 留出目标之前的序列（next-item 训练，不泄露 val/test 目标）
            if train_end >= min_train_len:
                train_seqs[user_id] = {
                    "sequence": seq[:train_end],
                    "behaviors": beh[:train_end],
                    "features": feat[:train_end],
                    "total_items": train_end,
                }

        logger.info(
            f"按目标行为({BEHAVIOR_NAMES[self.target_behavior]})切分: "
            f"train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}, "
            f"无目标行为用户={no_target_users}"
            + (" (已纳入训练)" if include_no_target else " (未纳入训练)")
        )
        return train_seqs, val_seqs, test_seqs

    def _save_split(self, sequences, products, item2idx, split_dir, tag):
        os.makedirs(split_dir, exist_ok=True)
        torch.save(sequences, os.path.join(split_dir, "sequences.pth"))
        torch.save(products, os.path.join(split_dir, "products.pth"))
        torch.save(item2idx, os.path.join(split_dir, "item2idx.pth"))
        torch.save(list(sequences.keys()), os.path.join(split_dir, "user_ids.pth"))
        logger.info(f"已保存 {tag} split -> {split_dir} ({len(sequences)} 个用户)")

    def _sample_users_and_save(self, full_train_seqs, products, item2idx, target_dir, ratio, seed=42):
        users = sorted(full_train_seqs.keys())
        rng = random.Random(seed)
        rng.shuffle(users)
        keep_n = max(1, int(round(len(users) * ratio)))
        sampled = {u: full_train_seqs[u] for u in users[:keep_n]}
        self._save_split(sampled, products, item2idx, target_dir, f"train_{ratio:.1f}")

    def _print_dataset_stats(self, sequences, products, train_seqs, val_seqs, test_seqs):
        divider = "=" * 60
        print(divider)
        print(f"数据集统计信息 - {self.config.dataset.name}")
        print(divider)
        print(f"  用户交互序列总数  : {len(sequences):>10,}")
        print(f"  商品总数          : {len(products):>10,}")
        # 行为分布
        beh_counter = defaultdict(int)
        for sd in sequences.values():
            for b in sd["behaviors"]:
                beh_counter[BEHAVIOR_NAMES[b]] += 1
        print("  行为分布:")
        for name in ["click", "like", "comment", "follow", "share", "favorite", "read"]:
            if beh_counter[name] > 0:
                print(f"    {name:<10}: {beh_counter[name]:>12,}")
        print(divider)
        print(f"  训练集 (train)    : {len(train_seqs):>10,} 条")
        print(f"  验证集 (val)      : {len(val_seqs):>10,} 条")
        print(f"  测试集 (test)     : {len(test_seqs):>10,} 条")
        print(divider)
