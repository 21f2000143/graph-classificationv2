"""
Phase 6 runner: Graph Pruning / Sparsification

Usage (from repo root):
  python scripts/run_pruning.py --model gat [--keep-ratios 0.25 0.5 0.75]

Loads the best-F1 GAT checkpoint and runs all three pruning strategies
on the test set, printing accuracy / F1 / inference speed comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    DROPOUT,
    GRAPH_DIR,
    HIDDEN_DIM,
    NUM_CLASSES,
    NUM_LAYERS,
    POOLING,
    RESULTS_DIR,
)
from src.dataset import load_splits
from src.models import get_model
from src.pruning import run_pruning_experiment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gat",
                   choices=["gcn", "sage", "gin", "gat"],
                   help="Model whose checkpoint to load.")
    p.add_argument("--keep-ratios", nargs="+", type=float,
                   default=[0.25, 0.50, 0.75])
    p.add_argument("--graph-dir", default=GRAPH_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(f"Phase 6 – Pruning Experiment  [{args.model.upper()}]  on {DEVICE}")
    print("=" * 60)

    # Load test split
    _, _, test_graphs = load_splits(args.graph_dir)
    in_channels = test_graphs[0].x.size(1)

    # Build and load model
    model = get_model(
        args.model,
        in_channels=in_channels,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        pooling=POOLING,
    )
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.model}_best_val_f1.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"[Model] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[Warning] Checkpoint not found at {ckpt_path}. Using random weights.")

    model = model.to(DEVICE)

    # Run all pruning strategies
    results = run_pruning_experiment(
        model, test_graphs, keep_ratios=tuple(args.keep_ratios)
    )

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{args.model}_pruning_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved → {result_path}")


if __name__ == "__main__":
    main()
