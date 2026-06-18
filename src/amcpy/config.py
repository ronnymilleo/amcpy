"""Project configuration and constants.

Replaces the old ``globals.py`` with a proper immutable config object
— no global mutable state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class Paths:
    """Filesystem paths used throughout the project.

    All directories are created automatically on first access via
    :meth:`ensure_dirs`.
    """

    root: Path = field(default_factory=lambda: Path(os.getcwd()))

    mat_data: Path = field(init=False)
    calculated_features: Path = field(init=False)
    arm_data: Path = field(init=False)
    trained_ann: Path = field(init=False)
    figures: Path = field(init=False)
    feature_figures: Path = field(init=False)

    mat_filename: str = "all_modulations.mat"

    def __post_init__(self) -> None:
        for name, sub in [
            ("mat_data", "mat-data"),
            ("calculated_features", "calculated-features"),
            ("arm_data", "arm-data"),
            ("trained_ann", "ann"),
            ("figures", "figures"),
            ("feature_figures", "figures/features"),
        ]:
            object.__setattr__(self, name, self.root / sub)

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for attr in (
            "mat_data",
            "calculated_features",
            "arm_data",
            "trained_ann",
            "figures",
            "feature_figures",
        ):
            p: Path = getattr(self, attr)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SignalConfig:
    """Modulation signal definitions and metadata."""

    modulations: tuple[str, ...] = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM")
    modulations_with_noise: tuple[str, ...] = (
        "BPSK",
        "QPSK",
        "8PSK",
        "16QAM",
        "64QAM",
        "WGN",
    )
    labels: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

    snr_values: dict[int, str] = field(
        default_factory=lambda: {
            0: "-10",
            1: "-8",
            2: "-6",
            3: "-4",
            4: "-2",
            5: "0",
            6: "2",
            7: "4",
            8: "6",
            9: "8",
            10: "10",
            11: "12",
            12: "14",
            13: "16",
            14: "18",
            15: "20",
        }
    )

    frame_size: int = 2048
    num_frames: int = 1000
    num_threads: int = 8

    # Mat-file variable names per modulation
    mat_info: dict[str, str] = field(
        default_factory=lambda: {
            "BPSK": "signal_bpsk",
            "QPSK": "signal_qpsk",
            "8PSK": "signal_8psk",
            "16QAM": "signal_qam16",
            "64QAM": "signal_qam64",
            "WGN": "signal_noise",
        }
    )


@dataclass(frozen=True)
class FeatureConfig:
    """Feature definitions — which features are available and used."""

    # LaTeX names for all 18 features
    names: ClassVar[dict[int, str]] = {
        1: r"$\gamma_{max}$",
        2: r"$\sigma_{ap}$",
        3: r"$\sigma_{dp}$",
        4: r"$\sigma_{aa}$",
        5: r"$\sigma_{af}$",
        6: r"$X$",
        7: r"$X_2$",
        8: r"$\mu_{42}^{a}$",
        9: r"$\mu_{42}^{f}$",
        10: r"$C_{20}$",
        11: r"$C_{21}$",
        12: r"$C_{40}$",
        13: r"$C_{41}$",
        14: r"$C_{42}$",
        15: r"$C_{60}$",
        16: r"$C_{61}$",
        17: r"$C_{62}$",
        18: r"$C_{63}$",
    }

    all_features: tuple[int, ...] = tuple(range(1, 19))  # 1..18
    used: tuple[int, ...] = (2, 4, 6, 8, 12, 14)

    @property
    def used_names(self) -> list[str]:
        return [self.names[f] for f in self.used]

    @property
    def num_used(self) -> int:
        return len(self.used)


@dataclass(frozen=True)
class TrainingConfig:
    """Neural network training hyperparameters."""

    training_snr: tuple[int, ...] = (10, 11, 12, 13, 14, 15)
    all_snr: tuple[int, ...] = tuple(range(16))
    plotting_snr: tuple[int, ...] = tuple(range(16))

    test_size: float = 0.2
    random_state: int = 42

    # Default NN hyperparameters
    activation: str = "relu"
    batch_size: int = 128
    dropout: float = 0.4
    epochs: int = 21
    learning_rate: float = 0.001418378071933655
    optimizer: str = "rmsprop"
    layer_size_hl1: int = 26
    layer_size_hl2: int = 29
    layer_size_hl3: int = 30

    # Features files naming pattern
    @property
    def feature_files(self) -> list[str]:
        return [f"{m}_features" for m in SignalConfig().modulations_with_noise]


@dataclass(frozen=True)
class Config:
    """Top-level immutable configuration."""

    paths: Paths = field(default_factory=Paths)
    signals: SignalConfig = field(default_factory=SignalConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
