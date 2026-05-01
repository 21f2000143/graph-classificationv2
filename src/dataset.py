"""
Phase 3 – Dataset Splits

Stratified train / validation / test splits (70 / 15 / 15) with
balanced cohort representation across splits.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import (
    GRAPH_DIR,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def _labels(graphs: List[Data]) -> np.ndarray:
    return np.array([g.y.item() for g in graphs])


def _cohorts(graphs: List[Data]) -> np.ndarray:
    return np.array([g.cohort for g in graphs])


def stratified_split(
    graphs: List[Data],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """Return (train, val, test) lists with stratified label distribution.

    Stratification key = label × cohort concatenated string so that both
    survival class and cohort origin are balanced across splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1."

    labels = _labels(graphs)
    cohorts = _cohorts(graphs)
    strat_key = np.array([f"{c}_{l}" for c, l in zip(cohorts, labels)])

    indices = np.arange(len(graphs))

    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio + test_ratio,
        random_state=random_seed,
    )
    train_idx, temp_idx = next(sss1.split(indices, strat_key))

    # Second split: val vs test from the temp set
    temp_strat = strat_key[temp_idx]
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed,
    )
    val_local_idx, test_local_idx = next(
        sss2.split(temp_idx, temp_strat)
    )
    val_idx = temp_idx[val_local_idx]
    test_idx = temp_idx[test_local_idx]

    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in val_idx]
    test = [graphs[i] for i in test_idx]

    _print_split_stats("Train", train)
    _print_split_stats("Val", val)
    _print_split_stats("Test", test)

    return train, val, test


def _print_split_stats(name: str, graphs: List[Data]) -> None:
    labels = _labels(graphs)
    cohorts = _cohorts(graphs)
    unique_labels, counts = np.unique(labels, return_counts=True)
    unique_cohorts, ccounts = np.unique(cohorts, return_counts=True)
    print(f"[{name}] {len(graphs)} samples | "
          f"Labels: { {int(l): int(c) for l,c in zip(unique_labels, counts)} } | "
          f"Cohorts: { {co: int(cc) for co, cc in zip(unique_cohorts, ccounts)} }")


def save_splits(train: List[Data], val: List[Data], test: List[Data],
                save_dir: str = GRAPH_DIR) -> None:
    """Save split lists to disk."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train, os.path.join(save_dir, "train.pt"))
    torch.save(val, os.path.join(save_dir, "val.pt"))
    torch.save(test, os.path.join(save_dir, "test.pt"))
    print(f"[Splits] Saved to {save_dir}")


def load_splits(save_dir: str = GRAPH_DIR
                ) -> Tuple[List[Data], List[Data], List[Data]]:
    """Load train/val/test splits from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = torch.load(os.path.join(save_dir, "train.pt"), map_location=device)
    val = torch.load(os.path.join(save_dir, "val.pt"), map_location=device)
    test = torch.load(os.path.join(save_dir, "test.pt"), map_location=device)
    return train, val, test
