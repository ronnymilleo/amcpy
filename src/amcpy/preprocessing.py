"""Data preprocessing — reshape, scale, and split feature matrices."""

from __future__ import annotations

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from amcpy.config import Config


def preprocess_data(
    cfg: Config,
    mode: str = "training",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load calculated features, standardise, and split into train/test.

    Parameters
    ----------
    cfg : Config
        Project configuration.
    mode : str
        ``"training"`` uses high-SNR data only;
        ``"test"`` uses all SNR levels.

    Returns
    -------
    x_train, x_test, y_train, y_test : np.ndarray
    scaler : StandardScaler
        Fitted scaler (for later use in evaluation).
    """
    s = cfg.signals
    t = cfg.training
    f = cfg.features

    snr_axis = t.training_snr if mode == "training" else t.all_snr

    n_samples = s.num_frames * len(snr_axis) * len(s.modulations_with_noise)
    x = np.zeros((n_samples, f.num_used), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    for mod_idx, mod in enumerate(t.feature_files):
        mod_name = mod.split("_")[0]
        print(f"Processing {mod_name} data ...")

        data = scipy.io.loadmat(str(cfg.paths.calculated_features / f"{mod}.mat"))
        mod_data = data[s.mat_info[mod_name]]

        base = mod_idx * s.num_frames * len(snr_axis)

        for snr_i, snr in enumerate(snr_axis):
            row_start = base + snr_i * s.num_frames
            row_end = row_start + s.num_frames
            x[row_start:row_end, :] = mod_data[snr, :, list(f.used)]

        y[base : base + s.num_frames * len(snr_axis)] = s.labels[mod_idx]

    # Standardise
    scaler = StandardScaler()
    scaler.fit(x)
    scaled_x = scaler.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        scaled_x,
        y,
        test_size=t.test_size,
        random_state=t.random_state,
        stratify=y,
    )

    print(
        f"\nData shape: {x_train.shape} {x_test.shape} {y_train.shape} {y_test.shape}"
    )
    return x_train, x_test, y_train, y_test, scaler
