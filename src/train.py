"""
Phase 5 – Training Utilities

- Training loop with early stopping (patience ≥ 10 epochs)
- Best-checkpoint saving (val loss AND val F1)
- Loss / F1 curve plotting
- ROC-AUC (one-vs-rest) curve plotting
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_CLASSES,
    PLOT_DIR,
    WEIGHT_DECAY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None,
                    num_classes: int = NUM_CLASSES) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1}

    if y_prob is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
            metrics["roc_auc_macro"] = auc
        except Exception:
            pass

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module) -> Tuple[float, Dict]:
    """Run one evaluation epoch.  Returns (loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    all_true, all_pred, all_prob = [], [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)
        total_loss += loss.item() * batch.num_graphs

        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = np.argmax(prob, axis=1)
        all_true.extend(batch.y.cpu().numpy().tolist())
        all_pred.extend(pred.tolist())
        all_prob.append(prob)

    avg_loss = total_loss / len(loader.dataset)
    all_prob = np.vstack(all_prob)
    metrics = compute_metrics(
        np.array(all_true), np.array(all_pred), all_prob
    )
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_graphs: List[Data],
    val_graphs: List[Data],
    model_name: str = "model",
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_dir: str = CHECKPOINT_DIR,
    plot_dir: str = PLOT_DIR,
) -> Dict:
    """Full training loop with early stopping and checkpoint saving.

    Returns history dict with train/val losses and metrics.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    train_loader = DataLoader(train_graphs, batch_size=batch_size,
                              shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_graphs, batch_size=batch_size,
                            shuffle=False, pin_memory=False)

    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    epochs_no_improve = 0

    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "train_acc": [], "val_acc": [],
    }

    for epoch in range(1, max_epochs + 1):
        # ── Train ──
        model.train()
        total_train_loss = 0.0
        all_true_tr, all_pred_tr = [], []

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_true_tr.extend(batch.y.cpu().numpy().tolist())
            all_pred_tr.extend(preds.tolist())

        train_loss = total_train_loss / len(train_loader.dataset)
        train_f1 = f1_score(all_true_tr, all_pred_tr,
                             average="macro", zero_division=0)
        train_acc = accuracy_score(all_true_tr, all_pred_tr)

        # ── Validate ──
        val_loss, val_metrics = evaluate(model, val_loader, criterion)
        val_f1 = val_metrics["f1_macro"]
        val_acc = val_metrics["accuracy"]

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f}  F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}  F1: {val_f1:.4f}"
        )

        # ── Checkpoint: best val loss ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir,
                                     f"{model_name}_best_val_loss.pt")
            torch.save(model.state_dict(), ckpt_path)

        # ── Checkpoint: best val F1 ──
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = os.path.join(checkpoint_dir,
                                     f"{model_name}_best_val_f1.pt")
            torch.save(model.state_dict(), ckpt_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ── Early stopping ──
        if epochs_no_improve >= patience:
            print(f"[Early Stopping] No F1 improvement for {patience} epochs.")
            break

    _plot_curves(history, model_name, plot_dir)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_curves(history: Dict, model_name: str, plot_dir: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"{model_name} – Loss Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{model_name}_loss.png"), dpi=150)
    plt.close(fig)

    # F1 curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_f1"], label="Train F1")
    ax.plot(epochs, history["val_f1"], label="Val F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"{model_name} – F1 Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{model_name}_f1.png"), dpi=150)
    plt.close(fig)

    print(f"[Plots] Saved loss/F1 curves → {plot_dir}")


def plot_roc_curves(model: nn.Module,
                    test_graphs: List[Data],
                    model_name: str,
                    plot_dir: str = PLOT_DIR,
                    num_classes: int = NUM_CLASSES,
                    batch_size: int = BATCH_SIZE) -> None:
    """Plot one-vs-rest ROC curves for all classes."""
    from sklearn.metrics import roc_curve, auc as sk_auc

    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    model.eval()
    all_true, all_prob = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch)
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            all_true.extend(batch.y.cpu().numpy().tolist())
            all_prob.append(prob)

    all_true = np.array(all_true)
    all_prob = np.vstack(all_prob)
    y_bin = label_binarize(all_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "darkorange", "green"]
    labels_str = ["Short (0)", "Mid (1)", "Long (2)"]

    for i, (color, lbl) in enumerate(zip(colors, labels_str)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_prob[:, i])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color,
                label=f"{lbl} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} – ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path = os.path.join(plot_dir, f"{model_name}_roc.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved ROC curve → {save_path}")
