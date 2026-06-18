"""Fixed-point quantization of neural network weights for ARM deployment.

Converts floating-point weights/biases into Q-format integers suitable
for microcontrollers with 16-bit fixed-point arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import torch

from amcpy.config import Config

# Q-format ranges (M integer bits, N fractional bits, total 16-bit)
_Q_RANGE: dict[str, tuple[float, float]] = {}
_RESOLUTION: list[float] = []

for _M in range(0, 7):
    _N = 15 - _M
    _key = f"Q{_M}.{_N}"
    _Q_RANGE[_key] = (-(2 ** (_M - 1)), 2 ** (_M - 1) - 2 ** (-_N))
    _RESOLUTION.append(2 ** (-_N))


def _find_best_q_format(min_val: float, max_val: float) -> str:
    """Return the narrowest Q-format that can represent [min_val, max_val]."""
    for key in [
        "Q0.15",
        "Q1.14",
        "Q2.13",
        "Q3.12",
        "Q4.11",
        "Q5.10",
        "Q6.9",
    ]:
        lo, hi = _Q_RANGE[key]
        if min_val >= lo and max_val <= hi:
            return key
    return "Q6.9"  # fallback — widest range


def _quantize_tensor(
    tensor: torch.Tensor,
    q_format: str,
) -> np.ndarray:
    """Quantize a float tensor to int16 using the given Q-format."""
    lo, hi = _Q_RANGE[q_format]
    scale = 2 ** int(q_format.split(".")[1])  # 2^N
    clamped = torch.clamp(tensor, lo, hi)
    quantized = torch.round(clamped * scale).to(torch.int16)
    return quantized.numpy()


def quantize(
    model: torch.nn.Module,
    sample_input: np.ndarray,
    cfg: Config,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Quantize all linear layers of a trained model.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    sample_input : np.ndarray
        Representative input data used to determine activation ranges.
    cfg : Config
        Project configuration.

    Returns
    -------
    save_dict : dict
        ``{"weights": ..., "biases": ...}`` — flattened int16 arrays.
    info_dict : dict
        Q-format strings for each layer's weights, biases, inputs, and outputs.
    """
    # Collect all Linear layers
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]

    # Determine weight/bias ranges
    max_w, max_b = [], []
    min_w, min_b = [], []
    for layer in linear_layers:
        w = layer.weight.data
        b = layer.bias.data
        max_w.append(float(w.max()))
        max_b.append(float(b.max()))
        min_w.append(float(w.min()))
        min_b.append(float(b.min()))

    # Best Q-format per layer
    info: dict[str, str] = {}
    for n in range(len(max_w)):
        info[f"Layer {n + 1} weights"] = _find_best_q_format(min_w[n], max_w[n])
        info[f"Layer {n + 1} biases"] = _find_best_q_format(min_b[n], max_b[n])

    # Input Q-format
    info["Input"] = _find_best_q_format(
        float(np.min(sample_input)), float(np.max(sample_input))
    )

    # Forward-pass to determine output ranges per layer
    x = torch.from_numpy(sample_input).float()
    with torch.no_grad():
        for n, layer in enumerate(linear_layers):
            x = layer(x)
            info[f"Layer {n + 1} outputs"] = _find_best_q_format(0.0, float(x.max()))

    # Quantize each layer
    quantized_weights = []
    quantized_biases = []
    for n, layer in enumerate(linear_layers):
        qw = info[f"Layer {n + 1} weights"]
        qb = info[f"Layer {n + 1} biases"]

        w_q = _quantize_tensor(layer.weight.data, qw)
        b_q = _quantize_tensor(layer.bias.data, qb)

        # Log quantization error
        w_deq = w_q.astype(np.float32) / (2 ** int(qw.split(".")[1]))
        b_deq = b_q.astype(np.float32) / (2 ** int(qb.split(".")[1]))
        print(
            f"Layer {n + 1} max weight error: "
            f"{float(np.max(np.abs(layer.weight.numpy() - w_deq))):.3g}"
        )
        print(
            f"Layer {n + 1} max bias error:   "
            f"{float(np.max(np.abs(layer.bias.numpy() - b_deq))):.3g}"
        )

        # Flatten and store
        quantized_weights.append(w_q.T.flatten())
        quantized_biases.append(b_q.flatten())

    all_weights = np.concatenate(quantized_weights)
    all_biases = np.concatenate(quantized_biases)

    save_dict = {"weights": all_weights, "biases": all_biases}
    scipy.io.savemat(
        str(cfg.paths.arm_data / "w_and_b.mat"),
        save_dict,
    )
    return save_dict, info
