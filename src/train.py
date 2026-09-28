"""
train.py

End-to-end training pipeline:
  1. Load data/dataset.csv (generate it first with data/generate_dataset.py)
  2. Split train/val/test, standardize features
  3. Train the DeepNeuralNetwork (src/model.py)
  4. Evaluate on the held-out test set (accuracy, precision, recall, F1, confusion matrix)
  5. Save the trained model, scaler, and metrics/plots for the dashboard to use

Usage:
    python src/train.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import FEATURE_NAMES  # noqa: E402
from model import DeepNeuralNetwork  # noqa: E402
from scaler import StandardScaler  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATASET_PATH = os.path.join(ROOT, "data", "dataset.csv")
MODELS_DIR = os.path.join(ROOT, "models")


def train_val_test_split(X, y, val_frac=0.15, test_frac=0.15, seed=42):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def evaluate(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def plot_history_and_confusion(history, metrics, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(history["loss"], label="train loss")
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="val loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks([0, 1], ["Benign", "Malicious"])
    axes[1].set_yticks([0, 1], ["Benign", "Malicious"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "training_report.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"No dataset found at {DATASET_PATH}. Generating one now...")
        sys.path.insert(0, os.path.join(ROOT, "data"))
        import generate_dataset
        generate_dataset.main()

    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df["label"].values.astype(np.float64)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    model = DeepNeuralNetwork(input_dim=X.shape[1], hidden_layers=(32, 16))
    history = model.fit(X_train_s, y_train, X_val_s, y_val, epochs=100, batch_size=64, lr=0.003, patience=10)

    y_pred = model.predict(X_test_s)
    metrics = evaluate(y_test, y_pred)

    print("\nTest set performance:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:10s}: {v:.4f}")
    print(f"  confusion   : {metrics['confusion_matrix']}")

    model.save(os.path.join(MODELS_DIR, "ids_model.npz"))
    scaler.save(os.path.join(MODELS_DIR, "scaler.npz"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "history": history, "feature_names": FEATURE_NAMES}, f, indent=2)

    plot_path = plot_history_and_confusion(history, metrics, MODELS_DIR)
    print(f"\nSaved model, scaler, metrics.json, and {os.path.basename(plot_path)} to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
