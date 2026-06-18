"""Visualisation of extracted features — mean curves, error bars, and multi-frame plots.

Uses matplotlib's built-in mathtext renderer (no LaTeX installation required).
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.io
from plotly.subplots import make_subplots

from amcpy.config import Config

# Use matplotlib's built-in math renderer (no LaTeX needed)
plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "dejavusans",
    }
)

_COLORS = ["#2F8000", "#DEAA0B", "#FF3300", "#AD00E6", "#0066FF"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data(cfg: Config) -> np.ndarray:
    """Load all modulation feature matrices into a stacked array.

    Returns
    -------
    np.ndarray
        Shape ``(n_modulations, n_snr, n_frames, n_used_features)``.
    """
    all_data = []
    for mod in cfg.signals.modulations:
        path = cfg.paths.calculated_features / f"{mod}_features.mat"
        f = scipy.io.loadmat(str(path))
        all_data.append(f[cfg.signals.mat_info[mod]][:, :, list(cfg.features.used)])
    return np.array(all_data)


def _compute_stats(data: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-SNR mean and standard deviation across frames."""
    n_mods = len(cfg.signals.modulations)
    n_snr = len(cfg.training.plotting_snr)
    n_ft = cfg.features.num_used
    mean = np.zeros((n_mods, n_snr, 1, n_ft))
    stddev = np.zeros_like(mean)
    for i in range(n_mods):
        for j in range(n_snr):
            for k in range(n_ft):
                mean[i, j, 0, k] = np.mean(data[i, j, :, k])
                stddev[i, j, 0, k] = np.std(data[i, j, :, k])
    return mean, stddev


def _snr_axis(cfg: Config) -> np.ndarray:
    """Generate SNR x-axis values."""
    snr_vals = np.linspace(-10, 20, len(cfg.training.plotting_snr))
    x = np.zeros((len(cfg.signals.modulations), len(cfg.training.plotting_snr)))
    for i in range(len(cfg.signals.modulations)):
        x[i, :] = snr_vals
    return x


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_means(
    snr_axis_vals: np.ndarray,
    data_axis: np.ndarray,
    cfg: Config,
    save: bool = True,
) -> None:
    """Plot mean feature curves per modulation (PNG)."""
    for n in range(cfg.features.num_used):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        for i, mod in enumerate(cfg.signals.modulations):
            plt.plot(
                snr_axis_vals[i, :],
                data_axis[i, :, 0, n],
                _COLORS[i],
                linewidth=1.0,
                antialiased=True,
            )
        plt.xlabel("SNR [dB]")
        plt.xticks(snr_axis_vals[0, :], cfg.signals.snr_values.values())
        plt.ylabel(cfg.features.used_names[n], rotation=0, fontsize=15, labelpad=20)
        plt.legend(cfg.signals.modulations)

        if save:
            fname = f"ft{cfg.features.used[n]}_mean.png"
            plt.savefig(
                cfg.paths.feature_figures / fname,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()


def plot_errorbars(
    snr_axis_vals: np.ndarray,
    mean: np.ndarray,
    stddev: np.ndarray,
    cfg: Config,
    save: bool = True,
) -> None:
    """Plot mean ± stddev for each feature."""
    for n in range(cfg.features.num_used):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        for i in range(len(cfg.signals.modulations)):
            plt.errorbar(
                snr_axis_vals[i, :],
                mean[i, :, 0, n],
                yerr=stddev[i, :, 0, n],
                color=_COLORS[i],
                linewidth=1.0,
            )
        plt.xlabel("SNR [dB]")
        plt.xticks(snr_axis_vals[0, :], cfg.signals.snr_values.values())
        plt.ylabel(cfg.features.used_names[n], rotation=0, fontsize=15, labelpad=20)
        plt.legend(cfg.signals.modulations)

        if save:
            fname = f"ft{cfg.features.used[n]}_err.png"
            plt.savefig(
                cfg.paths.feature_figures / fname,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()


def generate_html_plot(
    snr_axis_vals: np.ndarray,
    data_axis: np.ndarray,
    cfg: Config,
    save: bool = True,
) -> None:
    """Generate an interactive HTML plot with all features in subplots."""
    n_ft = cfg.features.num_used
    fig = make_subplots(
        rows=(n_ft + 4) // 5,
        cols=min(5, n_ft),
        subplot_titles=cfg.features.used_names,
    )
    rows, cols = 1, 1
    for ft in range(n_ft):
        if cols == 6:
            rows += 1
            cols = 1
        for label, signal in enumerate(cfg.signals.modulations):
            show = ft == 0
            fig.add_trace(
                go.Scatter(
                    x=snr_axis_vals[label, :],
                    y=data_axis[label, :, 0, ft],
                    legendgroup=signal,
                    name=signal,
                    showlegend=show,
                    line=dict(color=px.colors.qualitative.Plotly[label]),
                ),
                row=rows,
                col=cols,
            )
        cols += 1

    fig.update_layout(
        width=1920,
        height=1080,
        legend=dict(
            orientation="h",
            yanchor="auto",
            y=1.05,
            xanchor="auto",
            x=0,
            title="Modulation",
        ),
    )

    if save:
        fig.write_html(str(cfg.paths.feature_figures / "all_plots.html"))
    else:
        fig.show()


def run_plots(cfg: Config) -> None:
    """Execute all visualisation routines."""
    cfg.paths.ensure_dirs()
    data = _load_data(cfg)
    snr_arr = _snr_axis(cfg)
    mean_arr, stddev_arr = _compute_stats(data, cfg)

    plot_means(snr_arr, mean_arr, cfg, save=True)
    plot_errorbars(snr_arr, mean_arr, stddev_arr, cfg, save=True)
    generate_html_plot(snr_arr, mean_arr, cfg, save=True)

    print("All plots generated!")
