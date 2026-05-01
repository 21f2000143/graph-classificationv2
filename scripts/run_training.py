"""
Phase 4 + 5 runner: Model Training

Usage (from repo root):
  python scripts/run_training.py --model gcn
  python scripts/run_training.py --model gat --epochs 100 --batch-size 16

Supported models: gcn, sage, gin, gat
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch_geometric.loader import DataLoader as GeoLoader

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    GRAPH_DIR,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_CLASSES,
    NUM_LAYERS,
    PLOT_DIR,
    POOLING,
    RESULTS_DIR,
    WEIGHT_DECAY,
)
from src.dataset import load_splits
from src.models import get_model
from src.train import evaluate, plot_roc_curves, train_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gat",
                   choices=["gcn", "sage", "gin", "gat"],
                   help="GNN architecture to train.")
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--pooling", default=POOLING, choices=["mean", "sum", "max"])
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    p.add_argument("--graph-dir", default=GRAPH_DIR)
    p.add_argument("--no-roc", action="store_true",
                   help="Skip ROC curve plotting.")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(f"Phase 4+5 – Training  [{args.model.upper()}]  on {DEVICE}")
    print("=" * 60)

    # Load data
    train_graphs, val_graphs, test_graphs = load_splits(args.graph_dir)

    # Infer in_channels from first graph
    in_channels = train_graphs[0].x.size(1)
    print(f"[Data] Train={len(train_graphs)}  Val={len(val_graphs)}  "
          f"Test={len(test_graphs)}  in_channels={in_channels}")

    # Build model
    model_kwargs = dict(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    model = get_model(args.model, **model_kwargs)
    model = model.to(DEVICE)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    train_model(
        model=model,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        model_name=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=CHECKPOINT_DIR,
        plot_dir=PLOT_DIR,
    )

    # Final evaluation on test set (load best-F1 checkpoint)
    best_ckpt = os.path.join(CHECKPOINT_DIR, f"{args.model}_best_val_f1.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        print(f"\n[Eval] Loaded best-F1 checkpoint: {best_ckpt}")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics = evaluate(
        model,
        GeoLoader(test_graphs, batch_size=args.batch_size, shuffle=False),
        criterion,
    )

    print("\n── Test Results ──")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  test_loss: {test_loss:.4f}")

    # Detailed classification report
    loader = GeoLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    all_true, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_true.extend(batch.y.cpu().numpy())
            all_pred.extend(preds)

    print("\n── Classification Report ──")
    print(classification_report(
        all_true, all_pred,
        target_names=["Short (0)", "Mid (1)", "Long (2)"],
        zero_division=0,
    ))

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{args.model}_test_results.json")
    with open(result_path, "w") as f:
        json.dump({**test_metrics, "test_loss": test_loss}, f, indent=2)
    print(f"\n[Results] Saved → {result_path}")

    # ROC curves
    if not args.no_roc:
        plot_roc_curves(model, test_graphs, args.model, PLOT_DIR,
                        batch_size=args.batch_size)


if __name__ == "__main__":
    main()
