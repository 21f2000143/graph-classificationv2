"""
Phase 6 – Graph Pruning / Sparsification

Three pruning strategies:
1. **Attention-based edge dropping** — uses GAT attention weights to keep
   only the top-k% edges per graph.
2. **Degree-based node filtering** — removes low-degree nodes (and their
   incident edges) that carry less structural information.
3. **Random edge sparsification** — simple random baseline for comparison.

Evaluation: measures accuracy, F1, and inference time before/after pruning.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, subgraph

from src.config import BATCH_SIZE, DEVICE, NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attention-based edge pruning
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def attention_prune_graphs(
    model,
    graphs: List[Data],
    keep_ratio: float = 0.5,
    batch_size: int = BATCH_SIZE,
) -> List[Data]:
    """Return a list of pruned Data objects, keeping only the top
    `keep_ratio` edges (by mean attention weight across GAT heads).

    Args:
        model:       Trained GATClassifier.
        graphs:      List of Data objects.
        keep_ratio:  Fraction of edges to retain (per graph).
    """
    model.eval()
    pruned = []
    loader = DataLoader(graphs, batch_size=1, shuffle=False)

    for data in loader:
        data = data.to(DEVICE)
        try:
            _, attn_list = model(data.x, data.edge_index, data.batch,
                                 return_attention=True)
            # Use attention weights from the last GAT layer
            attn = attn_list[-1]           # [E, heads]
            attn_mean = attn.mean(dim=-1)  # [E]

            num_keep = max(1, int(keep_ratio * data.edge_index.size(1)))
            topk_idx = torch.topk(attn_mean, num_keep).indices

            new_edge_index = data.edge_index[:, topk_idx]
            pruned_data = Data(
                x=data.x,
                edge_index=new_edge_index,
                y=data.y,
            )
            pruned_data.cohort = getattr(data, "cohort", "unknown")
        except Exception:
            # Fall back to original graph if model doesn't support attention
            pruned_data = data.clone()

        pruned.append(pruned_data.cpu())

    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# 2. Degree-based node filtering
# ─────────────────────────────────────────────────────────────────────────────

def degree_prune_graphs(
    graphs: List[Data],
    min_degree: int = 1,
) -> List[Data]:
    """Remove nodes whose degree is below `min_degree`, along with their edges.

    Returns pruned Data list (nodes are re-indexed; labels preserved).
    """
    pruned = []
    for data in graphs:
        data = data.to("cpu")
        num_nodes = data.x.size(0)
        deg = degree(data.edge_index[0], num_nodes=num_nodes)

        keep_mask = deg >= min_degree
        keep_nodes = keep_mask.nonzero(as_tuple=False).view(-1)

        if keep_nodes.numel() == 0 or keep_nodes.numel() == num_nodes:
            pruned.append(data)
            continue

        new_edge_index, _ = subgraph(
            keep_nodes, data.edge_index, relabel_nodes=True,
            num_nodes=num_nodes
        )
        new_x = data.x[keep_nodes]

        pruned_data = Data(
            x=new_x.to(DEVICE),
            edge_index=new_edge_index.to(DEVICE),
            y=data.y.to(DEVICE),
        )
        pruned_data.cohort = getattr(data, "cohort", "unknown")
        pruned.append(pruned_data)

    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random edge sparsification (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def random_prune_graphs(
    graphs: List[Data],
    keep_ratio: float = 0.5,
    seed: int = 42,
) -> List[Data]:
    """Randomly retain `keep_ratio` fraction of edges per graph."""
    rng = np.random.default_rng(seed)
    pruned = []
    for data in graphs:
        data = data.to("cpu")
        num_edges = data.edge_index.size(1)
        num_keep = max(1, int(keep_ratio * num_edges))
        idx = rng.choice(num_edges, size=num_keep, replace=False)
        idx_t = torch.from_numpy(idx).long()

        pruned_data = Data(
            x=data.x.to(DEVICE),
            edge_index=data.edge_index[:, idx_t].to(DEVICE),
            y=data.y.to(DEVICE),
        )
        pruned_data.cohort = getattr(data, "cohort", "unknown")
        pruned.append(pruned_data)

    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_pruning(
    model,
    graphs: List[Data],
    label: str = "pruned",
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Evaluate model on a list of graphs; returns accuracy, F1, avg ms/graph."""
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_true, all_pred = [], []
    start = time.perf_counter()

    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.batch)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_true.extend(batch.y.cpu().numpy().tolist())
        all_pred.extend(preds.tolist())

    elapsed_ms = (time.perf_counter() - start) * 1000
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    ms_per_graph = elapsed_ms / max(len(graphs), 1)

    print(f"[{label}] Acc={acc:.4f} | F1={f1:.4f} | "
          f"Inference={elapsed_ms:.1f}ms total ({ms_per_graph:.3f} ms/graph)")
    return {"accuracy": acc, "f1_macro": f1,
            "total_ms": elapsed_ms, "ms_per_graph": ms_per_graph}


def run_pruning_experiment(
    model,
    test_graphs: List[Data],
    keep_ratios: Tuple[float, ...] = (0.25, 0.50, 0.75),
) -> dict:
    """Compare all three pruning strategies vs. unpruned baseline.

    Returns a dict of results keyed by strategy + keep_ratio.
    """
    results = {}

    # Baseline
    print("\n── Baseline (no pruning) ──")
    results["baseline"] = evaluate_pruning(model, test_graphs, "baseline")

    # Attention-based (requires GAT)
    print("\n── Attention-based edge pruning ──")
    for kr in keep_ratios:
        pruned = attention_prune_graphs(model, test_graphs, keep_ratio=kr)
        lbl = f"attn_keep{int(kr*100)}"
        results[lbl] = evaluate_pruning(model, pruned, lbl)

    # Degree-based
    print("\n── Degree-based node filtering ──")
    for min_deg in [1, 2, 3]:
        pruned = degree_prune_graphs(test_graphs, min_degree=min_deg)
        lbl = f"degree_min{min_deg}"
        results[lbl] = evaluate_pruning(model, pruned, lbl)

    # Random (baseline sparsification)
    print("\n── Random edge sparsification ──")
    for kr in keep_ratios:
        pruned = random_prune_graphs(test_graphs, keep_ratio=kr)
        lbl = f"random_keep{int(kr*100)}"
        results[lbl] = evaluate_pruning(model, pruned, lbl)

    return results
