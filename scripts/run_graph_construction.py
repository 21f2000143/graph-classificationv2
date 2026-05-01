"""
Phase 2 runner: Graph Construction

Usage (from repo root):
  python scripts/run_graph_construction.py [--cache-path PATH]

Builds PyG Data objects for all cohorts and saves:
  outputs/graphs/{tcga,sweden,metabric}/*.pt
  outputs/graphs/all_graphs.pt
  outputs/graphs/{train,val,test}.pt
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import GRAPH_DIR
from src.graph_construction import build_all_graphs
from src.dataset import stratified_split, save_splits
from src.preprocessing import run_preprocessing


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-path",
        default="outputs/graphs/preprocessed.pkl",
        help="Path to cached preprocessing artefacts (from run_preprocessing.py).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Phase 2 – Graph Construction")
    print("=" * 60)

    # Load or recompute preprocessing
    if os.path.exists(args.cache_path):
        print(f"[Cache] Loading from {args.cache_path} …")
        with open(args.cache_path, "rb") as f:
            preprocessed = pickle.load(f)
    else:
        print("[Cache] Not found — running preprocessing …")
        preprocessed = run_preprocessing()

    # Build graphs
    all_graphs = build_all_graphs(preprocessed, save_dir=GRAPH_DIR)

    # Stratified splits
    print("\n[Splits] Creating stratified train/val/test splits …")
    train, val, test = stratified_split(all_graphs)
    save_splits(train, val, test, save_dir=GRAPH_DIR)

    print(f"\n[Done] {len(all_graphs)} total graphs  "
          f"→ {len(train)} train / {len(val)} val / {len(test)} test")


if __name__ == "__main__":
    main()
