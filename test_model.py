"""
Chicago Crime GCN Model - Test Accuracy Script
================================================
Loads the trained GCN model and evaluates it on the test set.

Requirements:
    pip install torch torch-geometric scikit-learn pandas numpy

Run:
    python test_model.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROCESSED_DIR  = "processed"
OUTPUT_DIR     = "outputs"
MODEL_PATH     = f"{OUTPUT_DIR}/gcn_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# GCN ARCHITECTURE (same as model.py)
# ─────────────────────────────────────────────
class CrimeGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden_channels)
        self.conv2   = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_graph_data() -> tuple:
    """Load preprocessed graph data and return (data, node_df, label_encoder)."""
    print("\n[1/3] Loading preprocessed graph data...")

    node_df = pd.read_csv(f"{PROCESSED_DIR}/nodes.csv")
    edge_df = pd.read_csv(f"{PROCESSED_DIR}/edges.csv")

    print(f"  Nodes: {len(node_df):,}")
    print(f"  Edges: {len(edge_df):,}")

    # Node features
    X = torch.tensor(
        node_df[["lat_norm", "lon_norm", "crime_count_z", "log_crime_count"]].values,
        dtype=torch.float
    )

    # Labels
    le = LabelEncoder()
    y_raw = le.fit_transform(node_df["dominant_crime"].values)
    y = torch.tensor(y_raw, dtype=torch.long)
    num_classes = len(le.classes_)

    # Edge index
    src = torch.tensor(edge_df["src"].values, dtype=torch.long)
    dst = torch.tensor(edge_df["dst"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # Edge weights
    edge_weight = torch.tensor(edge_df["weight"].values, dtype=torch.float)

    # Create masks (same logic as training)
    n = len(node_df)
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.10
    TEST_RATIO = 0.20
    SEED = 42

    from sklearn.model_selection import train_test_split

    indices = np.arange(n)
    label_counts = pd.Series(y_raw).value_counts()
    can_stratify = (label_counts >= 2).all()
    stratify_arg = y_raw if can_stratify else None

    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - TRAIN_RATIO),
        random_state=SEED, stratify=stratify_arg
    )

    temp_labels = y_raw[temp_idx]
    temp_label_counts = pd.Series(temp_labels).value_counts()
    can_stratify_temp = (temp_label_counts >= 2).all()

    val_size_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_size_adj),
        random_state=SEED,
        stratify=temp_labels if can_stratify_temp else None
    )

    def make_mask(idx):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        return mask

    train_mask = make_mask(train_idx)
    val_mask = make_mask(val_idx)
    test_mask = make_mask(test_idx)

    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    print(f"  Classes: {num_classes}")
    print(f"  Crime types: {list(le.classes_)}")

    data = Data(
        x           = X,
        edge_index  = edge_index,
        edge_weight = edge_weight,
        y           = y,
        train_mask  = train_mask,
        val_mask    = val_mask,
        test_mask   = test_mask,
    )
    data.num_classes = num_classes
    data.label_encoder = le

    return data, node_df, le


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(data: Data):
    """Load trained model from checkpoint."""
    print(f"\n[2/3] Loading model from {MODEL_PATH}...")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    model = CrimeGCN(
        in_channels     = data.x.shape[1],
        hidden_channels = 128,
        out_channels    = data.num_classes,
        dropout         = 0.5,
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"  ✓ Model loaded successfully")

    return model


# ─────────────────────────────────────────────
# TEST & EVALUATE
# ─────────────────────────────────────────────
def test_model(model, data):
    """Test model on train, val, and test sets."""
    print(f"\n[3/3] Evaluating model on all splits...")

    model.eval()
    le = data.label_encoder

    with torch.no_grad():
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_weight)
        logits = out.cpu().numpy()
        true_labels = data.y.cpu().numpy()

    # Compute predictions
    preds = logits.argmax(axis=1)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    # Evaluate on each split
    splits = {
        "Train": data.train_mask.cpu().numpy(),
        "Val":   data.val_mask.cpu().numpy(),
        "Test":  data.test_mask.cpu().numpy(),
    }

    results = {}

    for split_name, mask in splits.items():
        y_true = true_labels[mask]
        y_pred = preds[mask]

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

        results[split_name] = {"accuracy": acc, "f1": f1, "y_true": y_true, "y_pred": y_pred}

        print(f"\n  ─────────────────────────────────────")
        print(f"  {split_name} Set Results")
        print(f"  ─────────────────────────────────────")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Macro F1: {f1:.4f}")

    # Detailed test set report
    print(f"\n  ─────────────────────────────────────")
    print(f"  Test Set - Classification Report")
    print(f"  ─────────────────────────────────────")

    y_test_true = true_labels[splits["Test"]]
    y_test_pred = preds[splits["Test"]]

    present_labels = np.unique(np.concatenate([y_test_true, y_test_pred]))
    present_names  = le.inverse_transform(present_labels)

    print(classification_report(
        y_test_true, y_test_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test_true, y_test_pred, labels=present_labels)

    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")

    return results, probs, logits, le


# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_results(results, le):
    """Generate accuracy and F1 comparison plot."""
    print(f"\n  Generating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    splits = list(results.keys())
    accuracies = [results[s]["accuracy"] for s in splits]
    f1_scores = [results[s]["f1"] for s in splits]

    # Accuracy
    axes[0].bar(splits, accuracies, color=["#2ecc71", "#3498db", "#e74c3c"], alpha=0.7)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy by Dataset Split")
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")

    # F1 Score
    axes[1].bar(splits, f1_scores, color=["#2ecc71", "#3498db", "#e74c3c"], alpha=0.7)
    axes[1].set_ylabel("Macro F1 Score")
    axes[1].set_title("F1 Score by Dataset Split")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/test_results.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ test_results.png saved")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Chicago Crime GCN Model - Test Accuracy")
    print("=" * 55)

    data, node_df, le = load_graph_data()
    model = load_model(data)
    results, probs, logits, le = test_model(model, data)
    plot_results(results, le)

    print("\n" + "=" * 55)
    print("  Testing complete!")
    print("=" * 55)


if __name__ == "__main__":
    main()
