"""
Phase 1 – Preprocessing & EDA

Responsibilities:
- Load mRNA-seq data for all three cohorts and transpose (genes→rows to
  samples→rows).
- Load clinical data; extract overall survival months/status.
- Load and filter the STRING PPI network (coexpression >= 190).
- Align gene features across all cohorts (intersection).
- Bin survival into three classes using data-driven percentile thresholds.
- Expose helper functions used by graph_construction.py.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import (
    COEXPRESSION_THRESHOLD,
    DAYS_PER_MONTH,
    DEVICE,
    METABRIC_CLINICAL_PATH,
    METABRIC_MRNA_PATH,
    PPI_PATH,
    SURVIVAL_BINS,
    SWEDEN_CLINICAL_PATH,
    SWEDEN_MRNA_PATH,
    TCGA_CLINICAL_PATH,
    TCGA_MRNA_PATH,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# PPI helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ppi(path: str = PPI_PATH,
             coexp_threshold: int = COEXPRESSION_THRESHOLD) -> pd.DataFrame:
    """Return filtered STRING PPI edge list (coexpression >= threshold)."""
    ppi = pd.read_csv(path)
    # Normalise column names – strip whitespace
    ppi.columns = [c.strip() for c in ppi.columns]
    ppi = ppi[ppi["coexpression"] >= coexp_threshold].reset_index(drop=True)
    print(f"[PPI] Loaded {len(ppi):,} edges with coexpression >= {coexp_threshold}.")
    return ppi


def build_gene_index(ppi: pd.DataFrame) -> Dict[str, int]:
    """Return a mapping gene_name -> integer index from PPI edge list."""
    genes = sorted(set(ppi["protein1"].tolist() + ppi["protein2"].tolist()))
    return {g: i for i, g in enumerate(genes)}


def build_edge_index(ppi: pd.DataFrame,
                     gene_index: Dict[str, int]) -> torch.Tensor:
    """Return edge_index tensor of shape [2, E] on CPU (moved to GPU later)."""
    src, dst = [], []
    for _, row in ppi.iterrows():
        g1, g2 = row["protein1"], row["protein2"]
        if g1 in gene_index and g2 in gene_index:
            src.append(gene_index[g1])
            dst.append(gene_index[g2])
            # Add reverse edge for undirected graph
            src.append(gene_index[g2])
            dst.append(gene_index[g1])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


# ─────────────────────────────────────────────────────────────────────────────
# mRNA-seq loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_mrna_txt(path: str) -> pd.DataFrame:
    """Load a tab-separated mRNA file (genes as rows, samples as columns)
    and transpose so that samples become rows and genes become columns."""
    df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    # Drop any non-numeric columns (e.g., Entrez_Gene_Id in TCGA/METABRIC)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.T  # samples × genes
    df.index.name = "sample_id"
    df.columns.name = "gene"
    return df


def _load_mrna_csv(path: str) -> pd.DataFrame:
    """Load a CSV mRNA file (genes as rows, samples as columns) and
    transpose so that samples become rows and genes become columns."""
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.T  # samples × genes
    df.index.name = "sample_id"
    df.columns.name = "gene"
    return df


def load_tcga(mrna_path: str = TCGA_MRNA_PATH,
              clin_path: str = TCGA_CLINICAL_PATH
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load TCGA-BRCA mRNA and clinical data."""
    mrna = _load_mrna_txt(mrna_path)
    clin = pd.read_csv(clin_path, sep="\t", low_memory=False)
    # Standardise column names
    clin.columns = [c.strip() for c in clin.columns]
    print(f"[TCGA] mRNA: {mrna.shape}  clinical: {clin.shape}")
    return mrna, clin


def load_sweden(mrna_path: str = SWEDEN_MRNA_PATH,
                clin_path: str = SWEDEN_CLINICAL_PATH
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Sweden (GSE96058) mRNA and clinical data.
    Converts survival days → months."""
    mrna = _load_mrna_csv(mrna_path)
    clin = pd.read_csv(clin_path, low_memory=False)
    clin.columns = [c.strip() for c in clin.columns]

    # Identify survival columns (case-insensitive)
    col_map = {c.lower(): c for c in clin.columns}
    # Days columns
    for day_col_key in ["overall survival days", "os_days", "os days",
                        "overall_survival_days"]:
        if day_col_key in col_map:
            orig = col_map[day_col_key]
            clin[orig] = pd.to_numeric(clin[orig], errors="coerce") / DAYS_PER_MONTH
            clin.rename(columns={orig: "Overall Survival (Months)"}, inplace=True)
            break

    for event_col_key in ["overall survival event", "os_event", "os event",
                          "overall_survival_event"]:
        if event_col_key in col_map:
            orig = col_map[event_col_key]
            clin.rename(columns={orig: "Overall Survival Status"}, inplace=True)
            break

    print(f"[Sweden] mRNA: {mrna.shape}  clinical: {clin.shape}")
    return mrna, clin


def load_metabric(mrna_path: str = METABRIC_MRNA_PATH,
                  clin_path: str = METABRIC_CLINICAL_PATH
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load METABRIC mRNA and clinical data."""
    mrna = _load_mrna_txt(mrna_path)
    clin = pd.read_csv(clin_path, sep="\t", low_memory=False)
    clin.columns = [c.strip() for c in clin.columns]
    print(f"[METABRIC] mRNA: {mrna.shape}  clinical: {clin.shape}")
    return mrna, clin


# ─────────────────────────────────────────────────────────────────────────────
# Clinical alignment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_survival(clin: pd.DataFrame,
                      sample_col: Optional[str] = None
                      ) -> pd.DataFrame:
    """Return a DataFrame with columns [sample_id, os_months, os_status].

    Attempts to find 'Overall Survival (Months)' and 'Overall Survival Status'
    in the clinical DataFrame by flexible name matching.
    """
    col_map = {c.lower().strip(): c for c in clin.columns}

    # Find os_months
    os_months_col = None
    for key in ["overall survival (months)", "os_months", "overall survival months",
                "os (months)", "overall_survival_months"]:
        if key in col_map:
            os_months_col = col_map[key]
            break

    # Find os_status
    os_status_col = None
    for key in ["overall survival status", "os_status", "overall survival event",
                "os_event", "overall_survival_status", "overall_survival_event"]:
        if key in col_map:
            os_status_col = col_map[key]
            break

    if os_months_col is None or os_status_col is None:
        raise ValueError(
            f"Cannot find survival columns.\n"
            f"Available: {list(clin.columns)}"
        )

    # Find sample ID column
    if sample_col is None:
        for key in ["patient id", "sample id", "sample_id", "#patient id",
                    "patient_id", "sampleid"]:
            if key in col_map:
                sample_col = col_map[key]
                break

    out = pd.DataFrame({
        "sample_id": clin[sample_col] if sample_col else clin.index,
        "os_months": pd.to_numeric(clin[os_months_col], errors="coerce"),
        "os_status": clin[os_status_col],
    }).dropna(subset=["os_months"])
    return out.reset_index(drop=True)


def compute_survival_thresholds(os_months: pd.Series) -> Tuple[float, float]:
    """Return (p33, p66) percentile thresholds for three-class binning."""
    p33 = float(np.percentile(os_months.dropna(), 33.33))
    p66 = float(np.percentile(os_months.dropna(), 66.67))
    print(f"[Binning] 33rd percentile = {p33:.2f}, 66th percentile = {p66:.2f} months")
    return p33, p66


def bin_survival(os_months: pd.Series,
                 p33: float,
                 p66: float) -> pd.Series:
    """Assign three-class survival label: 0 = short, 1 = mid, 2 = long."""
    labels = pd.cut(
        os_months,
        bins=[-np.inf, p33, p66, np.inf],
        labels=[0, 1, 2],
    )
    return labels.astype("Int64")


# ─────────────────────────────────────────────────────────────────────────────
# Gene-set alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_genes(mrna_frames: List[pd.DataFrame],
                ppi_genes: List[str]) -> List[pd.DataFrame]:
    """Restrict mRNA DataFrames to the intersection of PPI genes and
    the genes that appear in ALL cohorts.

    Returns the re-indexed DataFrames (same order, same columns).
    """
    # Intersect across cohorts and PPI
    common = set(ppi_genes)
    for df in mrna_frames:
        common &= set(df.columns.tolist())

    common_sorted = sorted(common)
    print(f"[Alignment] {len(common_sorted)} genes in intersection.")
    return [df[common_sorted] for df in mrna_frames], common_sorted


# ─────────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing() -> Dict:
    """End-to-end preprocessing.  Returns a dict with all processed artefacts
    needed by graph_construction.py."""

    # 1. PPI
    ppi = load_ppi()
    gene_index = build_gene_index(ppi)
    ppi_genes = list(gene_index.keys())
    edge_index = build_edge_index(ppi, gene_index)

    # 2. Load cohorts
    tcga_mrna, tcga_clin = load_tcga()
    sweden_mrna, sweden_clin = load_sweden()
    metabric_mrna, metabric_clin = load_metabric()

    # 3. Extract survival info
    tcga_surv = _extract_survival(tcga_clin)
    sweden_surv = _extract_survival(sweden_clin)
    metabric_surv = _extract_survival(metabric_clin)

    # 4. Align mRNA samples with clinical data
    def _align_mrna_clin(mrna: pd.DataFrame,
                         surv: pd.DataFrame) -> pd.DataFrame:
        """Inner join mRNA rows (index=sample_id) with survival table."""
        mrna = mrna.copy()
        mrna.index = mrna.index.astype(str).str.strip()
        surv = surv.copy()
        surv["sample_id"] = surv["sample_id"].astype(str).str.strip()
        merged = mrna.merge(
            surv.set_index("sample_id")[["os_months", "os_status"]],
            left_index=True, right_index=True, how="inner"
        )
        return merged

    tcga_df = _align_mrna_clin(tcga_mrna, tcga_surv)
    sweden_df = _align_mrna_clin(sweden_mrna, sweden_surv)
    metabric_df = _align_mrna_clin(metabric_mrna, metabric_surv)

    print(f"[Alignment] TCGA: {len(tcga_df)}, Sweden: {len(sweden_df)}, "
          f"METABRIC: {len(metabric_df)} samples after clinical join.")

    # 5. Gene alignment across cohorts
    mrna_cols = [
        df.drop(columns=["os_months", "os_status"]) for df in
        [tcga_df, sweden_df, metabric_df]
    ]
    aligned, common_genes = align_genes(mrna_cols, ppi_genes)
    tcga_df = pd.concat([aligned[0], tcga_df[["os_months", "os_status"]]], axis=1)
    sweden_df = pd.concat([aligned[1], sweden_df[["os_months", "os_status"]]], axis=1)
    metabric_df = pd.concat([aligned[2], metabric_df[["os_months", "os_status"]]], axis=1)

    # 6. Compute survival thresholds from combined distribution
    all_os = pd.concat([
        tcga_df["os_months"], sweden_df["os_months"], metabric_df["os_months"]
    ]).dropna()
    p33, p66 = compute_survival_thresholds(all_os)

    # 7. Bin survival labels
    for df in [tcga_df, sweden_df, metabric_df]:
        df["label"] = bin_survival(df["os_months"], p33, p66)

    # Drop samples with NaN label
    tcga_df = tcga_df.dropna(subset=["label"])
    sweden_df = sweden_df.dropna(subset=["label"])
    metabric_df = metabric_df.dropna(subset=["label"])

    print(f"[After binning] TCGA: {len(tcga_df)}, Sweden: {len(sweden_df)}, "
          f"METABRIC: {len(metabric_df)}")

    # 8. Rebuild gene_index restricted to common genes
    gene_index_common = {g: i for i, g in enumerate(common_genes)}
    edge_index_common = build_edge_index(ppi, gene_index_common)

    return {
        "ppi": ppi,
        "gene_index": gene_index_common,
        "common_genes": common_genes,
        "edge_index": edge_index_common,
        "tcga_df": tcga_df,
        "sweden_df": sweden_df,
        "metabric_df": metabric_df,
        "p33": p33,
        "p66": p66,
    }
