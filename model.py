"""
Chicago Crime Dataset - GCN Model
===================================
Step 2 of 4 in the Spatio-Temporal Crime Risk Prediction project.

What this script does:
  1. Loads preprocessed nodes.csv and edges.csv
  2. Builds a PyTorch Geometric graph Data object
  3. Defines a 2-layer GCN architecture (128 hidden units)
  4. Trains with stratified splits (70/20/10)
  5. Evaluates with Accuracy + Macro F1
  6. Saves the trained model + a crime hotspot heatmap

Requirements:
    pip install torch torch-geometric scikit-learn matplotlib pandas numpy tqdm

    PyTorch Geometric install (pick your CUDA version):
    https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

    Quick CPU-only install:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROCESSED_DIR  = "processed"
OUTPUT_DIR     = "outputs"
MODEL_PATH     = f"{OUTPUT_DIR}/gcn_model.pt"

HIDDEN_DIM     = 128
DROPOUT        = 0.5
LEARNING_RATE  = 0.01
WEIGHT_DECAY   = 5e-4
EPOCHS         = 300
PATIENCE       = 80
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.10
TEST_RATIO     = 0.20
SEED           = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# STEP 1: LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────
def load_graph_data() -> Data:
    print("\n[1/4] Loading preprocessed graph data...")

    node_df = pd.read_csv(f"{PROCESSED_DIR}/nodes.csv")
    edge_df = pd.read_csv(f"{PROCESSED_DIR}/edges.csv")

    print(f"  Nodes: {len(node_df):,}")
    print(f"  Edges: {len(edge_df):,}")

    # ── Node feature matrix X ──────────────────────────────────
    X = torch.tensor(
        node_df[["lat_norm", "lon_norm", "crime_count_z", "log_crime_count"]].values,
        dtype=torch.float
    )

    # ── Labels y ───────────────────────────────────────────────
    le = LabelEncoder()
    y_raw = le.fit_transform(node_df["dominant_crime"].values)
    y = torch.tensor(y_raw, dtype=torch.long)
    num_classes = len(le.classes_)
    print(f"  Classes (crime types): {num_classes}")
    print(f"  Class names: {list(le.classes_)}")

    # ── Edge index (COO format) ─────────────────────────────────
    src = torch.tensor(edge_df["src"].values, dtype=torch.long)
    dst = torch.tensor(edge_df["dst"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # ── Edge weights ───────────────────────────────────────────
    edge_weight = torch.tensor(edge_df["weight"].values, dtype=torch.float)

    # ── Train / Val / Test masks ───────────────────────────────
    n       = len(node_df)
    indices = np.arange(n)

    # Check if stratification is safe (every class needs >= 2 members)
    label_counts = pd.Series(y_raw).value_counts()
    can_stratify = (label_counts >= 2).all()

    if not can_stratify:
        small = label_counts[label_counts < 2].index.tolist()
        print(f"  ⚠ Skipping stratification — classes with <2 nodes: {small}")
        stratify_arg = None
    else:
        stratify_arg = y_raw

    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - TRAIN_RATIO),
        random_state=SEED, stratify=stratify_arg
    )

    # Second split: check stratification again on the temp subset
    temp_labels       = y_raw[temp_idx]
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
    val_mask   = make_mask(val_idx)
    test_mask  = make_mask(test_idx)

    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    data = Data(
        x           = X,
        edge_index  = edge_index,
        edge_weight = edge_weight,
        y           = y,
        train_mask  = train_mask,
        val_mask    = val_mask,
        test_mask   = test_mask,
    )
    data.num_classes   = num_classes
    data.label_encoder = le
    data.node_df       = node_df

    return data


# ─────────────────────────────────────────────
# STEP 2: GCN ARCHITECTURE
# ─────────────────────────────────────────────
class CrimeGCN(nn.Module):
    """
    2-layer Graph Convolutional Network for crime type classification.

    Architecture:
        Input (4 features) → GCNConv(128) → ReLU → Dropout
                           → GCNConv(num_classes) → log_softmax

    Deliberately shallow (2 layers) to:
      - Prevent over-smoothing on ~327 nodes
      - Keep localised spatial focus needed for hotspot detection
    """

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

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return layer-1 embeddings (useful for visualisation)."""
        x = self.conv1(x, edge_index, edge_weight)
        return F.relu(x)


# ─────────────────────────────────────────────
# STEP 3: TRAINING LOOP
# ─────────────────────────────────────────────
def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_model(data: Data):
    print("\n[2/4] Training GCN model...")

    model = CrimeGCN(
        in_channels     = data.x.shape[1],
        hidden_channels = HIDDEN_DIM,
        out_channels    = data.num_classes,
        dropout         = DROPOUT,
    ).to(device)

    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # ─────────────────────────────────────────────────────────────
    # CLASS WEIGHTS (Original formula - works better for this data)
    # ─────────────────────────────────────────────────────────────
    y_np = data.y.cpu().numpy()

    class_counts = np.bincount(y_np)
    class_weights = 1.0 / np.sqrt(class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index, data.edge_weight)
        loss = F.nll_loss(
            out[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights
        )
        loss.backward()
        optimizer.step()

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            out       = model(data.x, data.edge_index, data.edge_weight)
            val_loss  = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
            train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
            val_acc   = accuracy(out[data.val_mask],   data.y[data.val_mask])

        preds = out.argmax(dim=1).cpu().numpy()
        y_true = data.y.cpu().numpy()

        val_f1 = f1_score(
            y_true[data.val_mask.cpu()],
            preds[data.val_mask.cpu()],
            average="macro"
        )

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4} | Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                f"Val F1: {val_f1:.3f}")

    model.load_state_dict(best_state)
    return model, data, history


# ─────────────────────────────────────────────
# STEP 4: EVALUATION + OUTPUTS
# ─────────────────────────────────────────────
def evaluate(model, data):
    print("\n[3/4] Evaluating on test set...")

    model.eval()
    with torch.no_grad():
        out   = model(data.x, data.edge_index, data.edge_weight)
        preds = out[data.test_mask].argmax(dim=1).cpu().numpy()
        true  = data.y[data.test_mask].cpu().numpy()
        probs = torch.exp(out).cpu().numpy()

    acc = accuracy_score(true, preds)
    f1  = f1_score(true, preds, average="macro", zero_division=0)

    print(f"\n  ✓ Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ✓ Macro F1 Score: {f1:.4f}")
    print("\n  Per-class report:")
    present_labels = np.unique(np.concatenate([true, preds]))
    present_names  = data.label_encoder.inverse_transform(present_labels)
    print(classification_report(
        true, preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    return probs, preds


def save_plots(history, model, data, probs):
    print("\n[4/4] Saving plots...")

    # ── Training curves ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"],   label="Val Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
    plt.close()
    print(f"  ✓ training_curves.png saved")

    # ── Crime hotspot heatmaps ─────────────────────────────────
    node_df = data.node_df
    le      = data.label_encoder
    classes = list(le.classes_)

    # Pick top 2 most common classes to display
    top2 = classes[:2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, crime in zip(axes, top2):
        crime_idx = classes.index(crime)
        risk      = probs[:, crime_idx]

        sc = ax.scatter(
            node_df["lon"], node_df["lat"],
            c=risk, cmap="RdYlBu_r", s=80, alpha=0.85, edgecolors="none"
        )
        plt.colorbar(sc, ax=ax, label="Risk Probability")
        ax.set_title(f"Hotspot Map: {crime}", fontsize=13)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)

    plt.suptitle("Predicted Crime Risk Heatmaps — Chicago", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hotspot_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ hotspot_heatmaps.png saved")

    # ── Save node-level predictions for the web app ────────────
    node_df = node_df.copy()
    node_df["predicted_crime"] = le.inverse_transform(probs.argmax(axis=1))
    for i, cls in enumerate(classes):
        node_df[f"prob_{cls}"] = probs[:, i]

    node_df.to_csv(f"{OUTPUT_DIR}/node_predictions.csv", index=False)
    print(f"  ✓ node_predictions.csv saved  ← used by the React frontend")


def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n  ✓ Model saved → {MODEL_PATH}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Chicago Crime — GCN Model Training")
    print("=" * 55)

    data                 = load_graph_data()
    model, data, history = train_model(data)
    probs, preds         = evaluate(model, data)
    save_plots(history, model, data, probs)
    save_model(model)

    print("\n" + "=" * 55)
    print("  Training complete!")
    print(f"  Model   → {MODEL_PATH}")
    print(f"  Outputs → ./{OUTPUT_DIR}/")
    print("=" * 55)
    print("\nNext step: run  python app.py  to launch the web API")


if __name__ == "__main__":
    main()