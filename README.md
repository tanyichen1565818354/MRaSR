# MRaSR

Official implementation of **MRaSR: Multi-Behavior Relation-Augmented Sequential Recommendation**.

This repository is named `RDR4SR` for historical reasons (it extends DR4SR / RaSR). All experiment names and configs now refer to **MRaSR**.

MRaSR regenerates multi-behavior training sequences: it learns a behavior transition matrix \(\mathbf{R}\) to reweight item relations, protects high-intent interactions during augmentation, and decodes each step with a behavior-then-item dual-head generator. Downstream recommenders are unchanged.

## Requirements

Python 3.9+ and a CUDA GPU are recommended.

1. Install [PyTorch](https://pytorch.org/get-started/locally/) for your CUDA toolkit.
2. Install [pytorch-sparse](https://github.com/rusty1s/pytorch_sparse) for the **same** PyTorch / CUDA pair.
3. Install the remaining packages:

```bash
pip install -r requirements.txt
```

PLM embeddings use `sentence-transformers/all-mpnet-base-v2` via Hugging Face `transformers` (downloaded on first run).

Run every command from the repository root. Switch dataset / variant / backbone with Hydra overrides (`dataset=`, `graph_variant=`, `aug_variant=`, `regen_variant=`, `arch=`).

## Datasets

We use four Tenrec subsets: **QB-video**, **QK-article**, **QK-video**, and **SBR** (`sbr_data_1M`).
See [`data/README.md`](data/README.md) for download and directory layout.
Do not redistribute the raw CSVs.

Hydra dataset names: `qb_video`, `qk_article`, `qk_video`, `sbr_1m`.

Behavior ids are unified as
`click=0, like=1, comment=2, follow=3, share=4, favorite=5, read=6`.
The evaluation target is **like**.

## Reproduce the main table

Defaults are QB-video + full MRaSR. Default GPU is `cuda:0`.

### 1. Preprocess

```bash
python scripts/run_tenrec_preprocess.py
python scripts/run_tenrec_preprocess.py dataset=qk_article
```

### 2. Item PLM embeddings

```bash
python scripts/generate_plm_emb.py
python scripts/generate_plm_emb.py dataset=qk_article device=cuda:0
```

### 3. Relation graph with learnable \(\mathbf{R}\)

```bash
python scripts/build_graph.py
python scripts/build_graph.py dataset=qk_article
```

### 4. Behavior-aware augmentation

```bash
python scripts/run_augment.py
python scripts/run_augment.py dataset=qk_article
```

### 5. Train the original baseline and export item embeddings

Original training is required first: it reports the Original column and writes the checkpoint used by the regenerator.

```bash
python scripts/run_train.py data.use_original_only=true data.use_regen_augmented=false
python scripts/convert_embeddings.py model_name=sasbar dataset_name=QB-video
```

### 6. Train the regenerator and write regenerated sequences

```bash
python scripts/Pretrain_relation_regenerator.py
python scripts/generate_regenerated_data.py
```

### 7. Retrain the baseline on original + regenerated + augmented data (MRaSR)

```bash
python scripts/run_train.py
python scripts/run_train.py dataset=qk_video arch=grubar
```

`arch` is one of `sasbar`, `grubar`, `mbht`, `binn`, `rib`, `rlbl`.

## Ablations

Same scripts; only the variant override changes.

| Paper setting | Command extras |
|---------------|----------------|
| Full MRaSR | (defaults: `graph_variant=learned`, `aug_variant=learned`, `regen_variant=full`) |
| Predefined behavior weights | `graph_variant=static` then `aug_variant=static` then `regen_variant=static` |
| No dual-head decoder | `regen_variant=no_dual` |
| Behavior-agnostic augment | `aug_variant=uniform` then `regen_variant=uniform` |
| Flatten (item-only pipeline) | `graph_variant=flat` then `aug_variant=flat` then `regen_variant=flat` |

Example:

```bash
python scripts/build_graph.py dataset=qk_article graph_variant=flat
python scripts/run_augment.py dataset=qk_article aug_variant=flat
python scripts/Pretrain_relation_regenerator.py dataset=qk_article regen_variant=flat
```

## Data sparsity (30% / 70% users)

```bash
python scripts/split_mb_train_ratios.py --train-preprocessed data/QB-video/train/preprocessed
```

Then point `data.train_dir` at `data/QB-video/train_0.3/preprocessed` or `train_0.7/preprocessed`. Keep val/test unchanged.

## Single-behavior transfer

The SASRec / GRU4Rec / DR4SR / RaSR transfer experiments in the paper use the conference-code pipeline on like-only sequences. They are not part of this repository.

## Citation

If this code is useful, please cite the MRaSR paper and DR4SR.
