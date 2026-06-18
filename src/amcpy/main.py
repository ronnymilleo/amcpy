"""Main entry point for AMCPy — Automatic Modulation Classification in Python."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from amcpy.config import Config
from amcpy.feature_extraction import run_extraction
from amcpy.graphics import run_plots
from amcpy.nn_model import (
    AMCClassifier,
    confusion_matrix,
    evaluate_by_snr,
    load_model,
    train_model,
)
from amcpy.nn_quantization import quantize
from amcpy.preprocessing import preprocess_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AMCPy — Automatic Modulation Classification",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    sub.add_parser("extract", help="Extract features from raw .mat data")

    # plot
    sub.add_parser("plot", help="Generate feature visualisations")

    # train
    train_p = sub.add_parser("train", help="Train the neural network")
    train_p.add_argument("--epochs", type=int, default=21)
    train_p.add_argument("--batch-size", type=int, default=128)
    train_p.add_argument("--lr", type=float, default=0.001418)
    train_p.add_argument("--dropout", type=float, default=0.4)
    train_p.add_argument(
        "--optimizer", choices=["rmsprop", "adam", "nadam"], default="rmsprop"
    )
    train_p.add_argument("--activation", default="relu")

    # eval
    eval_p = sub.add_parser("eval", help="Evaluate a trained model")
    eval_p.add_argument("model_id", help="Model ID to evaluate")
    eval_p.add_argument(
        "--mode",
        choices=["training", "test"],
        default="test",
        help="training = high-SNR only; test = all SNR",
    )

    # quantize
    quant_p = sub.add_parser("quantize", help="Quantize model for ARM deployment")
    quant_p.add_argument("model_id", help="Model ID to quantize")

    # full
    sub.add_parser("full", help="Run full pipeline: extract → plot → train → eval")

    return parser


def _resolve_model_id(cfg: Config, model_id: str | None) -> str:
    """Resolve model ID — if not given, use the newest saved model."""
    if model_id:
        return model_id

    models = sorted(
        cfg.paths.trained_ann.glob("model-*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not models:
        raise FileNotFoundError(f"No models found in {cfg.paths.trained_ann}")

    newest = models[-1].stem.replace("model-", "")
    print(f"No model ID given — using newest: {newest}")
    return newest


def cmd_extract(cfg: Config) -> None:
    """Extract features from raw IQ data."""
    run_extraction(cfg)


def cmd_plot(cfg: Config) -> None:
    """Generate feature plots."""
    run_plots(cfg)


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    """Train the neural network."""
    cfg.paths.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train, x_test, y_train, y_test, scaler = preprocess_data(cfg, "training")

    s = cfg.signals
    f = cfg.features
    model = AMCClassifier(
        n_features=f.num_used,
        n_classes=len(s.modulations_with_noise),
        hl1=cfg.training.layer_size_hl1,
        hl2=cfg.training.layer_size_hl2,
        hl3=cfg.training.layer_size_hl3,
        dropout=args.dropout,
        activation=args.activation,
    )

    trained, model_id = train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        cfg,
        device,
    )

    evaluate_by_snr(trained, model_id, scaler, cfg, device)
    confusion_matrix(trained, model_id, x_test, y_test, cfg)


def cmd_eval(cfg: Config, args: argparse.Namespace) -> None:
    """Evaluate a saved model."""
    model_id = _resolve_model_id(cfg, args.model_id)
    model = load_model(model_id, cfg)

    _, x_test, _, y_test, scaler = preprocess_data(cfg, args.mode)

    evaluate_by_snr(model, model_id, scaler, cfg)
    confusion_matrix(model, model_id, x_test, y_test, cfg)


def cmd_quantize(cfg: Config, args: argparse.Namespace) -> None:
    """Quantize model weights for embedded deployment."""
    model_id = _resolve_model_id(cfg, args.model_id)
    model = load_model(model_id, cfg)

    x_train, x_test, _, _, _ = preprocess_data(cfg, "test")
    combined = np.concatenate([x_train, x_test])

    save_dict, info_dict = quantize(model, combined, cfg)
    for k, v in info_dict.items():
        print(f"  {k} → {v}")


def cmd_full(cfg: Config, args: argparse.Namespace) -> None:
    """Run the complete pipeline."""
    cmd_extract(cfg)
    cmd_plot(cfg)
    cmd_train(cfg, args)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = Config()
    cfg.paths.ensure_dirs()

    commands = {
        "extract": cmd_extract,
        "plot": cmd_plot,
        "train": cmd_train,
        "eval": cmd_eval,
        "quantize": cmd_quantize,
        "full": cmd_full,
    }

    commands[args.command](cfg, args)


if __name__ == "__main__":
    main()
