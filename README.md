# BL2403: Graph Neural Network-Based Breast Cancer Prognosis Using PPI Networks

This repository implements a full GNN pipeline for breast cancer prognosis using protein–protein interaction (PPI) network data and multi-cohort mRNA-seq expression data.

---

## Problem

Validate the prognostic relevance of four candidate genes — **IQGAP1**, **IQGAP2**, **FRG1**, **EEF1A2** — using STRING PPI network data and mRNA-seq expression data from three cohorts (TCGA-BRCA, Sweden/GSE96058, METABRIC), framed as a **graph classification problem** targeting overall survival.

---

## Project Structure

```
graph-classificationv2/
├── src/
│   ├── config.py                  # Paths, hyperparameters, device config
│   ├── preprocessing.py           # Phase 1: data loading, PPI filtering, gene alignment, survival binning
│   ├── graph_construction.py      # Phase 2: PyG Data object construction, serialisation
│   ├── dataset.py                 # Phase 3: stratified train/val/test splits
│   ├── models.py                  # Phase 4: GCN, GraphSAGE, GIN, GAT architectures
│   ├── train.py                   # Phase 5: training loop, early stopping, checkpoints, plots
│   └── pruning.py                 # Phase 6: attention/degree/random edge pruning
├── scripts/
│   ├── run_preprocessing.py       # Phase 1 runner (EDA + preprocessing)
│   ├── run_graph_construction.py  # Phase 2 runner (build & save graphs)
│   ├── run_training.py            # Phase 4+5 runner (train & evaluate)
│   └── run_pruning.py             # Phase 6 runner (pruning experiments)
├── notebooks/
│   ├── 01_eda.ipynb               # Interactive EDA notebook
│   └── 02_model_analysis.ipynb   # Model training, evaluation & pruning notebook
├── outputs/
│   ├── checkpoints/               # Saved model weights (.pt)
│   ├── graphs/                    # Serialised PyG graphs (.pt) + splits
│   ├── plots/                     # Loss/F1/ROC curves, EDA figures
│   └── results/                   # JSON test metrics
└── requirements.txt
```

---

## Environment

Use the project conda environment:

```bash
conda activate /mnt/kedargouri/sachin/pathomicfusion
```

If any packages are missing:

```bash
pip install -r requirements.txt
```

---

## Pipeline

### Phase 1 — Preprocessing & EDA

```bash
python scripts/run_preprocessing.py --save-plots
```

- Loads all three mRNA-seq cohorts and transposes matrices (genes → samples as rows)
- Filters STRING PPI edges by `coexpression >= 190`
- Aligns gene sets across cohorts (intersection)
- Bins overall survival into three classes using data-driven 33rd/66th percentile thresholds
- Generates EDA plots: survival distributions, class balance, candidate gene expression, PCA batch check

### Phase 2 — Graph Construction

```bash
python scripts/run_graph_construction.py
```

- Builds one PyG `Data` object per sample (nodes = genes, edges = co-expression, label = survival class)
- Saves individual `.pt` files per cohort plus a merged list
- Creates stratified 70/15/15 train/val/test splits

### Phase 3 — Model Training

```bash
# Train GAT (recommended — also supports pruning)
python scripts/run_training.py --model gat

# Other architectures
python scripts/run_training.py --model gcn
python scripts/run_training.py --model sage
python scripts/run_training.py --model gin
```

- Supports: GCN, GraphSAGE, GIN, GAT
- Early stopping (default patience = 15 epochs)
- Saves best-val-loss and best-val-F1 checkpoints
- Outputs: loss/F1 training curves, ROC-AUC curves, JSON test metrics

### Phase 4 — Graph Pruning

```bash
python scripts/run_pruning.py --model gat
```

Evaluates three pruning strategies:
1. **Attention-based edge dropping** (from GAT attention weights)
2. **Degree-based node filtering**
3. **Random edge sparsification** (baseline)

---

## Datasets

| Cohort    | mRNA-Seq | Clinical |
|-----------|----------|----------|
| TCGA-BRCA | `data_mrna_seq_v2_rsem_zscores_ref_diploid_samples_original.txt` | `brca_tcga_clinical_data.tsv` |
| Sweden    | `GSE96058_gene_expression_3273_samples_and_136_replicates_transformed_original.csv` | `GSE81540_parsed_clinical_features.csv` |
| METABRIC  | `data_mrna_illumina_microarray_zscores_ref_diploid_samples_original.txt` | `brca_metabric_clinical_data.tsv` |

PPI Network: STRING DB — filtered to edges with `coexpression >= 190`

---

## Survival Labelling

| Class | Label | Criterion |
|-------|-------|-----------|
| Short survival | `0` | < 33rd percentile (months) |
| Mid survival   | `1` | 33rd–66th percentile |
| Long survival  | `2` | > 66th percentile |

Thresholds are computed from the combined distribution of all three cohorts to ensure class balance.

---

## Model Architectures

All models share:
- N message-passing layers (default 3)
- BatchNorm + ReLU/ELU activations
- Dropout regularisation
- Global mean/sum/max pooling (configurable)
- Two-layer MLP classification head (3 output classes)

| Model | Key feature |
|-------|-------------|
| GCN | Baseline spectral convolution |
| GraphSAGE | Inductive; neighbourhood sampling |
| GIN | Maximum expressiveness (WL test equivalent) |
| GAT | Attention-weighted aggregation; exposes weights for pruning |

---

## Key Implementation Notes

- All computation uses GPU (`torch.device('cuda')`) when available
- Survival binning thresholds are data-driven (not hardcoded)
- Sweden cohort: survival days are converted to months (÷ 30.44)
- All mRNA-seq matrices are transposed before graph construction
- Only the `coexpression` channel from STRING DB is used for edge construction
