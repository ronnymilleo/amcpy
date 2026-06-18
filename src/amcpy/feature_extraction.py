"""Parallel feature extraction from raw modulation signal data.

Reads complex IQ samples from MATLAB ``.mat`` files, computes all 18 features
in parallel using multiple processes and threads, and saves per-modulation
feature matrices.
"""

from __future__ import annotations

import time
from multiprocessing import Process
from queue import Queue
from threading import Thread

import numpy as np
import scipy.io

from amcpy.config import Config
from amcpy.features import calculate_features


class _Worker(Thread):
    """Thread worker that pulls signal chunks from a queue and computes features."""

    def __init__(self, queue: Queue, feature_matrix: np.ndarray) -> None:
        super().__init__()
        self.queue = queue
        self.feature_matrix = feature_matrix

    def run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                signal, snr, frame = item
                self.feature_matrix[snr, frame, :] = calculate_features(
                    list(range(1, 19)), signal
                )
            finally:
                self.queue.task_done()


def _modulation_process(modulation: str, cfg: Config) -> None:
    """Extract features for a single modulation type (runs in its own process)."""
    print(f"[{modulation}] Starting feature extraction ...")

    mat_path = cfg.paths.mat_data / cfg.paths.mat_filename
    data_mat = scipy.io.loadmat(str(mat_path))
    parsed = data_mat[cfg.signals.mat_info[modulation]]

    # Thread pool
    queue: Queue = Queue()
    n_snr = len(cfg.signals.snr_values)
    n_frames = cfg.signals.num_frames
    n_all_features = len(cfg.features.all_features)

    features_matrix = np.zeros((n_snr, n_frames, n_all_features), dtype=np.float32)

    for _ in range(cfg.signals.num_threads):
        worker = _Worker(queue, features_matrix)
        worker.daemon = True
        worker.start()

    # Enqueue all work
    for snr in range(n_snr):
        for frame in range(n_frames):
            queue.put(
                [
                    parsed[snr, frame, 0 : cfg.signals.frame_size],
                    snr,
                    frame,
                ]
            )

    queue.join()

    # Save
    out_path = cfg.paths.calculated_features / f"{modulation}_features.mat"
    scipy.io.savemat(
        str(out_path),
        {"Modulation": modulation, cfg.signals.mat_info[modulation]: features_matrix},
    )
    print(f"[{modulation}] Done in {time.process_time():.1f}s → {out_path}")


def run_extraction(cfg: Config) -> None:
    """Run feature extraction for all modulation types in parallel."""
    cfg.paths.ensure_dirs()

    processes = []
    for mod in cfg.signals.modulations_with_noise:
        p = Process(target=_modulation_process, args=(mod, cfg))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All feature calculations complete!")
