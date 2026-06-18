"""Signal feature extraction for Automatic Modulation Classification.

Implements 18 statistical features based on instantaneous signal properties
(amplitude, phase, frequency) and higher-order cumulants.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Instantaneous signal values
# ---------------------------------------------------------------------------


class InstantaneousValues:
    """Compute instantaneous amplitude, phase, frequency and CN amplitude."""

    def __init__(self, signal: np.ndarray) -> None:
        """
        Parameters
        ----------
        signal : np.ndarray
            Complex-valued input signal.
        """
        self.abs: np.ndarray = np.abs(signal)
        self.phase: np.ndarray = np.angle(signal)
        self.unwrapped_phase: np.ndarray = np.unwrap(self.phase)
        self.frequency: np.ndarray = np.diff(self.unwrapped_phase) / (2 * np.pi)
        self.cn_amplitude: np.ndarray = self.abs / np.mean(self.abs) - 1


# ---------------------------------------------------------------------------
# Higher-order moments
# ---------------------------------------------------------------------------


class MomentValues:
    """Compute mixed moments M_{pq} = E[x^{p-q} · (x*)^{q}] of a signal."""

    def __init__(self, signal: np.ndarray) -> None:
        x = signal
        xc = np.conj(x)

        self.m20: complex = np.mean(x**2)
        self.m21: float = np.mean(x * xc).real
        self.m22: complex = np.mean(xc**2)

        self.m40: complex = np.mean(x**4)
        self.m41: complex = np.mean(x**3 * xc)
        self.m42: float = np.mean(x**2 * xc**2).real
        self.m43: complex = np.mean(x * xc**3)

        self.m60: complex = np.mean(x**6)
        self.m61: complex = np.mean(x**5 * xc)
        self.m62: float = np.mean(x**4 * xc**2).real
        self.m63: complex = np.mean(x**3 * xc**3)


# ---------------------------------------------------------------------------
# Individual feature functions (1–18)
# ---------------------------------------------------------------------------


def _gmax(signal: np.ndarray) -> float:
    """Feature 1: maximum of the squared spectral power density."""
    fft_abs = np.abs(np.fft.fft(signal))
    return float(np.max(fft_abs**2 / len(signal)))


def _std_abs_phase(signal: np.ndarray) -> float:
    """Feature 2: standard deviation of absolute instantaneous phase."""
    return float(np.std(np.abs(np.angle(signal)), ddof=1))


def _std_direct_phase(signal: np.ndarray) -> float:
    """Feature 3: standard deviation of direct (non-absolute) instantaneous phase."""
    return float(np.std(np.angle(signal), ddof=1))


def _std_abs_cna(signal: np.ndarray) -> float:
    """Feature 4: standard deviation of absolute centred-normalised amplitude."""
    iv = InstantaneousValues(signal)
    return float(np.std(np.abs(iv.cn_amplitude), ddof=1))


def _std_cnf(signal: np.ndarray) -> float:
    """Feature 5: standard deviation of instantaneous frequency."""
    iv = InstantaneousValues(signal)
    return float(np.std(iv.frequency, ddof=1))


def _mean_magnitude(signal: np.ndarray) -> float:
    """Feature 6: mean of the signal magnitude."""
    return float(np.mean(np.abs(signal)))


def _norm_sqrt_sum_amp(signal: np.ndarray) -> float:
    """Feature 7: normalised square root of the sum of amplitudes."""
    return float(np.sqrt(np.sum(np.abs(signal))) / len(signal))


def _kurtosis_cna(signal: np.ndarray) -> float:
    """Feature 8: kurtosis of the centred-normalised amplitude."""
    iv = InstantaneousValues(signal)
    return float(stats.kurtosis(iv.cn_amplitude, fisher=False))


def _kurtosis_cnf(signal: np.ndarray) -> float:
    """Feature 9: kurtosis of the instantaneous frequency."""
    iv = InstantaneousValues(signal)
    return float(stats.kurtosis(iv.frequency, fisher=False))


def _cumulant_20(signal: np.ndarray) -> float:
    """Feature 10: |C₂₀|."""
    return float(np.abs(MomentValues(signal).m20))


def _cumulant_21(signal: np.ndarray) -> float:
    """Feature 11: |C₂₁|."""
    return float(np.abs(MomentValues(signal).m21))


def _cumulant_40(signal: np.ndarray) -> float:
    """Feature 12: |C₄₀| = |m₄₀ − 3·m₂₀²|."""
    m = MomentValues(signal)
    return float(np.abs(m.m40 - 3 * m.m20**2))


def _cumulant_41(signal: np.ndarray) -> float:
    """Feature 13: |C₄₁| = |m₄₁ − 3·m₂₀·m₂₁|."""
    m = MomentValues(signal)
    return float(np.abs(m.m41 - 3 * m.m20 * m.m21))


def _cumulant_42(signal: np.ndarray) -> float:
    """Feature 14: |C₄₂| = |m₄₂ − |m₂₀|² − 2·m₂₁²|."""
    m = MomentValues(signal)
    return float(np.abs(m.m42 - np.abs(m.m20) ** 2 - 2 * m.m21**2))


def _cumulant_60(signal: np.ndarray) -> float:
    """Feature 15: |C₆₀| = |m₆₀ − 15·m₂₀·m₄₀ + 3·m₂₀³|."""
    m = MomentValues(signal)
    return float(np.abs(m.m60 - 15 * m.m20 * m.m40 + 3 * m.m20**3))


def _cumulant_61(signal: np.ndarray) -> float:
    """Feature 16: |C₆₁|."""
    m = MomentValues(signal)
    return float(
        np.abs(m.m61 - 5 * m.m21 * m.m40 - 10 * m.m20 * m.m41 + 30 * m.m20**2 * m.m21)
    )


def _cumulant_62(signal: np.ndarray) -> float:
    """Feature 17: |C₆₂|."""
    m = MomentValues(signal)
    return float(
        np.abs(
            m.m62
            - 6 * m.m20 * m.m42
            - 8 * m.m21 * m.m41
            - m.m22 * m.m40
            + 6 * m.m20**2 * m.m22
            + 24 * m.m21**2 * m.m20
        )
    )


def _cumulant_63(signal: np.ndarray) -> float:
    """Feature 18: |C₆₃|."""
    m = MomentValues(signal)
    return float(
        np.abs(
            m.m63
            - 9 * m.m21 * m.m42
            + 12 * m.m21**3
            - 3 * m.m20 * m.m43
            - 3 * m.m22 * m.m41
            + 18 * m.m20 * m.m21 * m.m22
        )
    )


# ---------------------------------------------------------------------------
# Feature dispatch table
# ---------------------------------------------------------------------------

_FEATURE_FUNCTIONS: dict[int, callable] = {
    1: _gmax,
    2: _std_abs_phase,
    3: _std_direct_phase,
    4: _std_abs_cna,
    5: _std_cnf,
    6: _mean_magnitude,
    7: _norm_sqrt_sum_amp,
    8: _kurtosis_cna,
    9: _kurtosis_cnf,
    10: _cumulant_20,
    11: _cumulant_21,
    12: _cumulant_40,
    13: _cumulant_41,
    14: _cumulant_42,
    15: _cumulant_60,
    16: _cumulant_61,
    17: _cumulant_62,
    18: _cumulant_63,
}


def calculate_features(
    feature_ids: list[int],
    signal: np.ndarray,
) -> list[float]:
    """Compute a list of features for a given complex signal.

    Parameters
    ----------
    feature_ids : list[int]
        Which features to compute (1–18).
    signal : np.ndarray
        Complex-valued input signal.

    Returns
    -------
    list[float]
        Computed feature values, in the same order as *feature_ids*.
    """
    return [_FEATURE_FUNCTIONS[fid](signal) for fid in feature_ids]


# ---------------------------------------------------------------------------
# Unit tests (run with pytest)
# ---------------------------------------------------------------------------


def _test_signal() -> np.ndarray:
    """Deterministic test signal used across all feature tests."""
    return np.array(
        [
            0 + 0j,
            -1 + 1j,
            2 - 2j,
            -3 + 3j,
            4 - 4j,
            -5 + 5j,
            6 - 6j,
            -7 + 7j,
            8 - 8j,
            -9 + 9j,
        ]
    )


def test_instantaneous_values():
    sig = _test_signal()
    iv = InstantaneousValues(sig)

    assert len(iv.abs) == 10
    assert len(iv.phase) == 10
    assert len(iv.unwrapped_phase) == 10
    assert len(iv.frequency) == 9
    assert len(iv.cn_amplitude) == 10

    # Spot checks
    assert np.isclose(iv.abs[1], np.sqrt(2), atol=1e-10)
    assert np.isclose(iv.cn_amplitude[0], -1.0, atol=1e-10)
    assert np.isclose(iv.cn_amplitude[-1], 1.0, atol=1e-10)


def test_moment_values():
    sig = _test_signal()
    m = MomentValues(sig)

    assert np.isclose(m.m21, 57.0, atol=1e-10)
    assert np.isclose(m.m42, 6133.2, atol=1e-6)
    assert np.isclose(m.m63, 782724.0, atol=1e-6)


def test_all_features():
    sig = _test_signal()

    expected = {
        1: 405.0,
        2: 0.940293603578649,
        3: 1.5903100728408748,
        4: 0.3312693299999689,
        5: 0.5153882032022075,
        6: 6.363961030678928,
        7: 0.7977443845417482,
        8: 1.7757575757575754,
        9: 1.0627162629757787,
        10: 57.0,
        11: 57.0,
        12: 3613.8,
        13: 3613.8,
        14: 3613.8,
        15: 3905583.0,
        16: 1094628.0,
        17: 311904.0,
        18: 1094628.0,
    }

    for fid, exp in expected.items():
        result = _FEATURE_FUNCTIONS[fid](sig)
        assert np.isclose(result, exp, rtol=1e-5), (
            f"Feature {fid}: expected {exp}, got {result}"
        )
