# Data layout

Do **not** upload Tenrec raw files or processed `.pth` tensors with the code.
Apply for Tenrec at [https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec/tenrec.html](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec/tenrec.html) and place the CSVs as follows:

```
data/Tenrec/Tenrec/QB-video.csv
data/Tenrec/Tenrec/QK-article.csv
data/Tenrec/Tenrec/QK-video.csv
data/Tenrec/Tenrec/sbr_data_1M.csv
```

After `scripts/run_tenrec_preprocess.py`, each dataset looks like:

```
data/<dataset>/train/preprocessed/{sequences,products,item2idx}.pth
data/<dataset>/val/preprocessed/...
data/<dataset>/test/preprocessed/...
```

`<dataset>` is one of `QB-video`, `QK-article`, `QK-video`, `sbr_1m`.

Later stages write under the same train split:

| Directory | Produced by |
|-----------|-------------|
| `train/preprocessed/embeddings/` | `scripts/generate_plm_emb.py` |
| `train/graph_learned/` | `scripts/build_graph.py` (`*_learned` configs) |
| `train/augment_learned/` | `scripts/run_augment.py` (`*_learned` configs) |
| `train/embeddings/` | `scripts/convert_embeddings.py` |
| `train/regeneration/<baseline>/` | `scripts/generate_regenerated_data.py` |
| `train_0.3/`, `train_0.7/` | `scripts/split_mb_train_ratios.py` |
