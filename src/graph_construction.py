"""
Phase 2 – Graph Construction

Responsibilities:
- Build one PyG `Data` object per sample per cohort.
- Node features: expression values of the aligned genes.
- Edges: co-expression edges from the filtered STRING DB.
- Label: three-class survival label (0/1/2).
- Save individual graphs as .pt files and a merged list.
- All tensors are placed on DEVICE (GPU when available).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.config import DEVICE, GRAPH_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Single graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(expr_values: np.ndarray,
                edge_index: torch.Tensor,
                label: int,
                sample_id: str,
                cohort: str) -> Data:
    """Build a PyG Data object for a single sample.

    Args:
        expr_values: 1-D array of expression values, one per node/gene.
        edge_index:  LongTensor [2, E] on CPU – shared across all samples.
        label:       Integer survival class {0, 1, 2}.
        sample_id:   String identifier of the sample.
        cohort:      Cohort name ('tcga' | 'sweden' | 'metabric').

    Returns:
        PyG Data object on DEVICE.
    """
    # Node features: shape [N, 1]
    x = torch.tensor(expr_values, dtype=torch.float32).unsqueeze(1)
    # Replace any NaN/Inf with 0
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    y = torch.tensor([label], dtype=torch.long)

    data = Data(
        x=x.to(DEVICE),
        edge_index=edge_index.to(DEVICE),
        y=y.to(DEVICE),
    )
    data.sample_id = sample_id
    data.cohort = cohort
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Cohort graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cohort_graphs(df: pd.DataFrame,
                        common_genes: List[str],
                        edge_index: torch.Tensor,
                        cohort: str,
                        save_dir: Optional[str] = None) -> List[Data]:
    """Build PyG graphs for all samples in a cohort DataFrame.

    Args:
        df:          DataFrame with gene columns + 'label' column.
                     Index is sample_id.
        common_genes: Ordered list of gene names (defines node ordering).
        edge_index:  Shared edge_index tensor.
        cohort:      Cohort name string.
        save_dir:    If given, save each graph as <save_dir>/<cohort>_<idx>.pt

    Returns:
        List of PyG Data objects.
    """
    gene_cols = common_genes  # preserves order
    graphs = []

    for i, (sample_id, row) in enumerate(df.iterrows()):
        expr = row[gene_cols].values.astype(np.float32)
        label = int(row["label"])
        g = build_graph(expr, edge_index, label, str(sample_id), cohort)
        graphs.append(g)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{cohort}_{i:06d}.pt")
            torch.save(g, save_path)

        if (i + 1) % 500 == 0:
            print(f"  [{cohort}] Built {i+1}/{len(df)} graphs …")

    print(f"[{cohort}] Total graphs built: {len(graphs)}")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_all_graphs(preprocessed: Dict,
                     save_dir: str = GRAPH_DIR) -> List[Data]:
    """Build graphs for all three cohorts and return merged list.

    Args:
        preprocessed: Dict returned by preprocessing.run_preprocessing().
        save_dir:     Directory to save individual .pt graph files.

    Returns:
        Merged list of Data objects (all cohorts).
    """
    edge_index = preprocessed["edge_index"]
    common_genes = preprocessed["common_genes"]

    cohort_map = {
        "tcga": preprocessed["tcga_df"],
        "sweden": preprocessed["sweden_df"],
        "metabric": preprocessed["metabric_df"],
    }

    all_graphs: List[Data] = []
    for cohort, df in cohort_map.items():
        print(f"\n[Graph Construction] Building {cohort} graphs ({len(df)} samples)…")
        graphs = build_cohort_graphs(
            df, common_genes, edge_index, cohort,
            save_dir=os.path.join(save_dir, cohort)
        )
        all_graphs.extend(graphs)

    # Save merged list
    merged_path = os.path.join(save_dir, "all_graphs.pt")
    torch.save(all_graphs, merged_path)
    print(f"\n[Graph Construction] Saved {len(all_graphs)} graphs → {merged_path}")

    return all_graphs


def load_graphs(save_dir: str = GRAPH_DIR) -> List[Data]:
    """Load pre-built graphs from disk (merged list)."""
    merged_path = os.path.join(save_dir, "all_graphs.pt")
    if not os.path.exists(merged_path):
        raise FileNotFoundError(
            f"Merged graph file not found at {merged_path}. "
            "Run build_all_graphs() first."
        )
    graphs = torch.load(merged_path, map_location=DEVICE)
    print(f"[Load] Loaded {len(graphs)} graphs from {merged_path}")
    return graphs
