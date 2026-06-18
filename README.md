# AMCPy — Automatic Modulation Classification in Python

Automatic classification of digital modulation schemes (BPSK, QPSK, 8PSK, 16QAM, 64QAM, WGN)
using statistical signal features and a PyTorch neural network.

## Features

- **18 statistical features**: spectral max, instantaneous phase/amplitude/frequency statistics,
  higher-order cumulants (C₂₀ through C₆₃), and more
- **Parallel feature extraction**: multi-process + multi-thread for fast processing of large datasets
- **PyTorch neural network**: fully-connected classifier with configurable architecture
- **Per-SNR evaluation**: accuracy curves for each modulation across all SNR levels
- **Fixed-point quantization**: Q-format quantization for ARM microcontroller deployment
- **Interactive visualisations**: Plotly HTML plots, matplotlib PNGs, confusion matrices, error bars
- **Weights & Biases integration**: experiment tracking and hyperparameter sweeps

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Install
pip install -e .

# Run the full pipeline (requires .mat data in mat-data/)
amcpy full

# Or step by step
amcpy extract      # Extract features from raw IQ samples
amcpy plot         # Generate feature visualisations
amcpy train        # Train the neural network
amcpy eval <id>    # Evaluate a saved model
amcpy quantize <id> # Quantize weights for embedded deployment
```

## Project Structure

```
amcpy/
├── config.py              # Immutable configuration (paths, signals, features, training)
├── features.py            # 18 statistical feature functions + unit tests
├── feature_extraction.py  # Parallel feature extraction from .mat files
├── preprocessing.py       # Data loading, standardisation, train/test split
├── nn_model.py            # PyTorch classifier, training, evaluation
├── nn_quantization.py     # Fixed-point Q-format quantization
├── graphics.py            # Feature visualisation (PNG + HTML)
├── main.py                # CLI entry point
└── __init__.py

mat-data/                  # Place your all_modulations.mat here
calculated-features/       # Output: per-modulation feature .mat files
ann/                       # Output: trained model checkpoints
figures/                   # Output: plots and confusion matrices
arm-data/                  # Output: quantized weights for microcontroller
```

## Data Format

The pipeline expects a MATLAB file `mat-data/all_modulations.mat` with variables:

| Variable | Modulation |
|----------|-----------|
| `signal_bpsk` | BPSK |
| `signal_qpsk` | QPSK |
| `signal_8psk` | 8PSK |
| `signal_qam16` | 16QAM |
| `signal_qam64` | 64QAM |
| `signal_noise` | WGN |

Each variable should be a 3D array of shape `(n_snr, n_frames, frame_size)` with complex IQ samples.

## Configuration

All settings are in `amcpy/config.py` using frozen dataclasses:

- `SignalConfig`: modulation names, SNR values, frame size, number of frames
- `FeatureConfig`: which of the 18 features to use (default: [2, 4, 6, 8, 12, 14])
- `TrainingConfig`: neural network hyperparameters, train/test split
- `Paths`: output directory locations

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- See `pyproject.toml` for full dependency list
- No LaTeX installation required (uses matplotlib's built-in mathtext)

## License

MIT — see [LICENSE](LICENSE)
