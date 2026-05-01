"""
Phase 1 runner: Preprocessing & EDA

Usage (from repo root, using the project conda env):
  python scripts/run_preprocessing.py [--save-plots]

Outputs:
  - outputs/plots/eda_*  (expression, survival, class-balance figures)
  - outputs/graphs/preprocessed.pkl  (optional cache)
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.config import CANDIDATE_GENES, PLOT_DIR
from src.preprocessing import run_preprocessing


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-plots", action="store_true",
                   help="Save EDA plots to outputs/plots/")
    p.add_argument("--cache-path", default="outputs/graphs/preprocessed.pkl",
                   help="Where to cache the preprocessed artefacts.")
    return p.parse_args()


def save_eda_plots(preprocessed: dict, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    tcga = preprocessed["tcga_df"]
    sweden = preprocessed["sweden_df"]
    metabric = preprocessed["metabric_df"]

    # ── 1. Survival distribution (all cohorts) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, (name, df) in zip(axes, [("TCGA", tcga), ("Sweden", sweden),
                                      ("METABRIC", metabric)]):
        ax.hist(df["os_months"].dropna(), bins=40, edgecolor="black", alpha=0.7)
        ax.set_title(f"{name} – OS Months")
        ax.set_xlabel("Months")
        ax.set_ylabel("Count")
    fig.suptitle("Overall Survival Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "eda_survival_dist.png"), dpi=150)
    plt.close(fig)

    # ── 2. Class balance per cohort ──
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, df) in zip(axes, [("TCGA", tcga), ("Sweden", sweden),
                                      ("METABRIC", metabric)]):
        counts = df["label"].value_counts().sort_index()
        ax.bar([str(int(c)) for c in counts.index], counts.values,
               color=["steelblue", "darkorange", "green"])
        ax.set_title(f"{name} – Class Balance")
        ax.set_xlabel("Survival Class (0=Short, 1=Mid, 2=Long)")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "eda_class_balance.png"), dpi=150)
    plt.close(fig)

    # ── 3. Expression distribution for candidate genes ──
    common = preprocessed["common_genes"]
    candidate_present = [g for g in CANDIDATE_GENES if g in common]

    if candidate_present:
        fig, axes = plt.subplots(1, len(candidate_present),
                                 figsize=(4 * len(candidate_present), 4))
        if len(candidate_present) == 1:
            axes = [axes]
        for ax, gene in zip(axes, candidate_present):
            for name, df in [("TCGA", tcga), ("Sweden", sweden),
                              ("METABRIC", metabric)]:
                if gene in df.columns:
                    ax.hist(df[gene].dropna(), bins=30, alpha=0.5, label=name)
            ax.set_title(gene)
            ax.set_xlabel("Expression (z-score)")
            ax.legend(fontsize=7)
        fig.suptitle("Candidate Gene Expression Distributions")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "eda_candidate_genes.png"), dpi=150)
        plt.close(fig)
        print(f"[EDA] Candidate genes present in alignment: {candidate_present}")
    else:
        print("[EDA] Warning: None of the candidate genes found in the common gene set.")

    # ── 4. PCA batch-effect check ──
    all_expr = pd.concat([
        tcga.drop(columns=["os_months", "os_status", "label"]),
        sweden.drop(columns=["os_months", "os_status", "label"]),
        metabric.drop(columns=["os_months", "os_status", "label"]),
    ], axis=0).fillna(0)
    cohort_labels = (
        ["TCGA"] * len(tcga) +
        ["Sweden"] * len(sweden) +
        ["METABRIC"] * len(metabric)
    )

    # Subsample for speed
    if len(all_expr) > 2000:
        idx = np.random.choice(len(all_expr), 2000, replace=False)
        all_expr = all_expr.iloc[idx]
        cohort_labels = [cohort_labels[i] for i in idx]

    pca = PCA(n_components=2)
    pc = pca.fit_transform(all_expr.values)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors_map = {"TCGA": "steelblue", "Sweden": "darkorange", "METABRIC": "green"}
    for cohort in set(cohort_labels):
        mask = [c == cohort for c in cohort_labels]
        ax.scatter(pc[mask, 0], pc[mask, 1],
                   label=cohort, alpha=0.4, s=8,
                   color=colors_map.get(cohort, "grey"))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA – Batch Effect Check Across Cohorts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "eda_pca_cohorts.png"), dpi=150)
    plt.close(fig)

    print(f"[EDA] All plots saved to {plot_dir}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Phase 1 – Preprocessing & EDA")
    print("=" * 60)

    preprocessed = run_preprocessing()

    # Summary statistics
    for name, df in [("TCGA", preprocessed["tcga_df"]),
                     ("Sweden", preprocessed["sweden_df"]),
                     ("METABRIC", preprocessed["metabric_df"])]:
        print(f"\n[{name}] {len(df)} samples | "
              f"Class distribution: "
              f"{ df['label'].value_counts().sort_index().to_dict() }")

    print(f"\n[Genes] {len(preprocessed['common_genes'])} common genes used as nodes.")
    print(f"[PPI]   {preprocessed['edge_index'].size(1)//2} undirected edges "
          f"(stored as {preprocessed['edge_index'].size(1)} directed).")

    if args.save_plots:
        save_eda_plots(preprocessed, PLOT_DIR)

    # Cache preprocessed artefacts
    cache_path = args.cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        # edge_index needs to be on CPU for pickling
        save_dict = {k: v for k, v in preprocessed.items()}
        save_dict["edge_index"] = preprocessed["edge_index"].cpu()
        pickle.dump(save_dict, f)
    print(f"\n[Cache] Saved preprocessed artefacts → {cache_path}")


if __name__ == "__main__":
    main()
