"""
Centralized configuration for the BL2403 GNN pipeline.
"""

import os
import torch

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Data paths
# ──────────────────────────────────────────────
PPI_PATH = (
    "/mnt/kedargouri/sachin/projects/bl2403-network-analysis-approach/"
    "data/filtered_genes.csv"
)

TCGA_MRNA_PATH = (
    "/home/sachin/projects/bl2403v2/data/tcga_cbioportal/"
    "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples_original.txt"
)
TCGA_CLINICAL_PATH = (
    "/home/sachin/projects/bl2403v2/data/tcga_cbioportal/"
    "brca_tcga_clinical_data.tsv"
)

SWEDEN_MRNA_PATH = (
    "/home/sachin/projects/bl2403v2/data/sweden/"
    "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed_original.csv"
)
SWEDEN_CLINICAL_PATH = (
    "/home/sachin/projects/bl2403v2/data/sweden/"
    "GSE81540_parsed_clinical_features.csv"
)

METABRIC_MRNA_PATH = (
    "/home/sachin/projects/bl2403v2/data/METABRIC/brca_metabric/"
    "data_mrna_illumina_microarray_zscores_ref_diploid_samples_original.txt"
)
METABRIC_CLINICAL_PATH = (
    "/home/sachin/projects/bl2403v2/data/METABRIC/"
    "brca_metabric_clinical_data.tsv"
)

# ──────────────────────────────────────────────
# Output directories
# ──────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

for _d in [GRAPH_DIR, CHECKPOINT_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ──────────────────────────────────────────────
# PPI / graph construction
# ──────────────────────────────────────────────
COEXPRESSION_THRESHOLD = 190   # filter STRING edges

# Candidate genes of interest
CANDIDATE_GENES = ["IQGAP1", "IQGAP2", "FRG1", "EEF1A2"]

# ──────────────────────────────────────────────
# Survival binning
# ──────────────────────────────────────────────
# Empirically determined from percentile analysis (see preprocessing.py)
# Values below are defaults; they are overridden at runtime from data.
SURVIVAL_BINS = [0, 30, 60, float("inf")]   # < 30 | 30–60 | > 60 months
SURVIVAL_LABELS = [0, 1, 2]

# Days-to-months conversion factor (Sweden cohort)
DAYS_PER_MONTH = 30.44

# ──────────────────────────────────────────────
# Dataset splits
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Model hyperparameters
# ──────────────────────────────────────────────
NUM_CLASSES = 3
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
POOLING = "mean"       # "mean" | "sum" | "max"

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
