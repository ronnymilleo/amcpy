"""Neural network model definition and training for AMC.

PyTorch-based replacement for the original TensorFlow/Keras implementation.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from amcpy.config import Config

# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------


class AMCClassifier(nn.Module):
    """Fully-connected classifier for Automatic Modulation Classification.

    Architecture (matching the original TF model):
        Input → Dense(n_features, ReLU) → Dropout →
                Dense(hl1, ReLU) → Dropout →
                Dense(hl2, ReLU) → Dropout →
                Dense(hl3, ReLU) → Dropout →
                Dense(n_classes, Softmax)
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hl1: int = 26,
        hl2: int = 29,
        hl3: int = 30,
        dropout: float = 0.4,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        act_cls: type[nn.Module] = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }.get(activation, nn.ReLU)

        self.layers = nn.Sequential(
            nn.Linear(n_features, hl1),
            nn.BatchNorm1d(hl1),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hl1, hl2),
            nn.BatchNorm1d(hl2),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hl2, hl3),
            nn.BatchNorm1d(hl3),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hl3, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _to_tensor(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(dtype)


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
    device: torch.device | None = None,
) -> tuple[nn.Module, str]:
    """Train the AMC classifier.

    Returns the trained model and its unique ID.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_cfg = cfg.training
    model_id = str(uuid.uuid4()).split("-")[0]
    model = model.to(device)

    x_train_t = _to_tensor(x_train).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    x_test_t = _to_tensor(x_test).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    # Optimiser selection
    lr = t_cfg.learning_rate
    if t_cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif t_cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    batch_size = t_cfg.batch_size
    n_samples = x_train_t.shape[0]

    history: dict[str, list[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(t_cfg.epochs):
        # --- Training ---
        model.train()
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_train_t[idx], y_train_t[idx]

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            correct += (pred.argmax(1) == yb).sum().item()
            total += xb.size(0)

        history["loss"].append(epoch_loss / total)
        history["accuracy"].append(correct / total)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_pred = model(x_test_t)
            val_loss = criterion(val_pred, y_test_t).item()
            val_acc = (val_pred.argmax(1) == y_test_t).float().mean().item()

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1:3d}/{t_cfg.epochs} | "
            f"loss: {history['loss'][-1]:.4f} | "
            f"acc: {history['accuracy'][-1]:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

    # Save model
    save_path = cfg.paths.trained_ann / f"model-{model_id}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_id": model_id,
            "config": cfg,
        },
        save_path,
    )
    print(f"\nModel saved → {save_path}")

    # Quick evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(x_train_t)
        train_acc = (train_pred.argmax(1) == y_train_t).float().mean().item()
    print(f"Training accuracy: {train_acc:.4f}")

    # Confusion matrix
    _plot_confusion_matrix(model, x_test_t, y_test_t, model_id, cfg)
    _plot_history(history, model_id, cfg)

    return model, model_id


def load_model(model_id: str, cfg: Config) -> AMCClassifier:
    """Load a saved model by ID."""
    path = cfg.paths.trained_ann / f"model-{model_id}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    s = cfg.signals
    f = cfg.features
    model = AMCClassifier(
        n_features=f.num_used,
        n_classes=len(s.modulations_with_noise),
        hl1=cfg.training.layer_size_hl1,
        hl2=cfg.training.layer_size_hl2,
        hl3=cfg.training.layer_size_hl3,
        dropout=cfg.training.dropout,
        activation=cfg.training.activation,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_by_snr(
    model: nn.Module,
    model_id: str,
    scaler,
    cfg: Config,
    device: torch.device | None = None,
) -> np.ndarray:
    """Evaluate accuracy per modulation per SNR level."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s = cfg.signals
    t = cfg.training
    model = model.to(device)
    model.eval()

    result = np.zeros((len(s.modulations_with_noise), len(t.all_snr)))

    for i, mod in enumerate(t.feature_files):
        print(f"Evaluating {mod.split('_')[0]} ...")
        data = scipy.io.loadmat(str(cfg.paths.calculated_features / f"{mod}.mat"))
        mod_data = data[s.mat_info[mod.split("_")[0]]]

        for snr in t.all_snr:
            x = mod_data[snr, :, list(cfg.features.used)]
            x = scaler.transform(x)
            x_t = _to_tensor(x).to(device)

            with torch.no_grad():
                preds = model(x_t).argmax(1).cpu().numpy()

            true_label = s.labels[i]
            result[i][snr] = accuracy_score([true_label] * len(preds), preds)

    _plot_accuracy_by_snr(result, model_id, cfg)

    scipy.io.savemat(
        str(cfg.paths.figures / f"{model_id}_figure_data.mat"),
        {"acc": result},
    )
    return result


def confusion_matrix(
    model: nn.Module,
    model_id: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
) -> None:
    """Generate and save a confusion matrix."""
    device = next(model.parameters()).device
    x_t = _to_tensor(x_test).to(device)
    y_t = torch.tensor(y_test, dtype=torch.long).to(device)
    _plot_confusion_matrix(model, x_t, y_t, model_id, cfg)


# ---------------------------------------------------------------------------
# Internal plotting helpers
# ---------------------------------------------------------------------------


def _plot_confusion_matrix(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_id: str,
    cfg: Config,
) -> None:
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(1).cpu().numpy()
    y_np = y.cpu().numpy()

    n_classes = len(cfg.signals.modulations_with_noise)
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_np, preds):
        cm[t, p] += 1
    cm_norm = np.around(cm / cm.sum(axis=1, keepdims=True), decimals=2)

    df = pd.DataFrame(
        cm_norm,
        index=cfg.signals.modulations_with_noise,
        columns=cfg.signals.modulations_with_noise,
    )

    plt.figure(figsize=(8, 4), dpi=150)
    sns.heatmap(df, annot=True, cmap=plt.cm.get_cmap("Blues", 6))
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.savefig(
        cfg.paths.figures / f"cm-{model_id}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def _plot_history(
    history: dict[str, list[float]],
    model_id: str,
    cfg: Config,
) -> None:
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["accuracy"], label="Train")
    ax1.plot(epochs, history["val_accuracy"], label="Test")
    ax1.set_title("Model accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(loc="best")

    ax2.plot(epochs, history["loss"], label="Train")
    ax2.plot(epochs, history["val_loss"], label="Test")
    ax2.set_title("Model loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(
        cfg.paths.figures / f"history-{model_id}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def _plot_accuracy_by_snr(
    result: np.ndarray,
    model_id: str,
    cfg: Config,
) -> None:
    plt.figure(figsize=(6, 3), dpi=300)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("SNR [dB]")
    plt.xticks(
        range(len(cfg.training.all_snr)),
        [cfg.signals.snr_values[i] for i in cfg.training.all_snr],
    )
    for i, mod in enumerate(cfg.signals.modulations_with_noise):
        plt.plot(result[i] * 100, label=mod)
    plt.legend(loc="best")
    plt.savefig(
        cfg.paths.figures / f"accuracy-{model_id}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
