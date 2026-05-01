"""
Phase 4 – GNN Model Architectures

Four graph classification models:
  - GCN      (baseline)
  - GraphSAGE (inductive, suitable for varying graph sizes)
  - GIN      (stronger expressiveness)
  - GAT      (attention-weighted aggregation)

Each model uses:
  - {NUM_LAYERS} message-passing layers
  - Global pooling (mean / sum / max)
  - MLP classification head
  - Dropout for regularisation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from src.config import DROPOUT, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, POOLING


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_pool_fn(pooling: str):
    return {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool,
    }[pooling]


class _MLPHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 dropout: float = DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
# GCN
# ─────────────────────────────────────────────────────────────────────────────

class GCNClassifier(nn.Module):
    """Graph Convolutional Network for graph-level classification."""

    def __init__(self,
                 in_channels: int = 1,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT,
                 pooling: str = POOLING):
        super().__init__()
        self.pool_fn = _get_pool_fn(pooling)
        self.dropout = dropout

        dims = [in_channels] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.head = _MLPHead(hidden_dim, hidden_dim // 2, num_classes, dropout)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool_fn(x, batch)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE
# ─────────────────────────────────────────────────────────────────────────────

class SAGEClassifier(nn.Module):
    """GraphSAGE for graph-level classification."""

    def __init__(self,
                 in_channels: int = 1,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT,
                 pooling: str = POOLING):
        super().__init__()
        self.pool_fn = _get_pool_fn(pooling)
        self.dropout = dropout

        dims = [in_channels] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.head = _MLPHead(hidden_dim, hidden_dim // 2, num_classes, dropout)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool_fn(x, batch)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# GIN
# ─────────────────────────────────────────────────────────────────────────────

class GINClassifier(nn.Module):
    """Graph Isomorphism Network for graph-level classification."""

    def __init__(self,
                 in_channels: int = 1,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT,
                 pooling: str = POOLING):
        super().__init__()
        self.pool_fn = _get_pool_fn(pooling)
        self.dropout = dropout

        dims = [in_channels] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Linear(dims[i + 1], dims[i + 1]),
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        self.head = _MLPHead(hidden_dim, hidden_dim // 2, num_classes, dropout)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool_fn(x, batch)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# GAT
# ─────────────────────────────────────────────────────────────────────────────

class GATClassifier(nn.Module):
    """Graph Attention Network for graph-level classification.

    Exposes attention weights for downstream pruning.
    """

    def __init__(self,
                 in_channels: int = 1,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 num_classes: int = NUM_CLASSES,
                 heads: int = 4,
                 dropout: float = DROPOUT,
                 pooling: str = POOLING):
        super().__init__()
        self.pool_fn = _get_pool_fn(pooling)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            out_ch = hidden_dim // heads
            self.convs.append(
                GATConv(in_ch, out_ch, heads=heads,
                        dropout=dropout, concat=True)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = _MLPHead(hidden_dim, hidden_dim // 2, num_classes, dropout)

    def forward(self, x, edge_index, batch, return_attention: bool = False):
        attn_weights = []
        for conv, bn in zip(self.convs, self.bns):
            if return_attention:
                x, (_, attn) = conv(
                    x, edge_index, return_attention_weights=True
                )
                attn_weights.append(attn)
            else:
                x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool_fn(x, batch)
        logits = self.head(x)

        if return_attention:
            return logits, attn_weights
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gcn": GCNClassifier,
    "sage": SAGEClassifier,
    "gin": GINClassifier,
    "gat": GATClassifier,
}


def get_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
