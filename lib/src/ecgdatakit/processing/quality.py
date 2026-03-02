"""ECG signal quality assessment."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead
from ecgdatakit.processing._core import require_scipy
from ecgdatakit.processing.peaks import detect_r_peaks


def signal_quality_index(lead: Lead) -> float:
    """Compute a composite signal quality index (SQI) in the range [0, 1].

    Combines four sub-metrics: kurtosis SQI, power-ratio SQI,
    R-peak regularity SQI, and baseline stability SQI.

    Parameters
    ----------
    lead : Lead
        Input ECG lead.

    Returns
    -------
    float
        Score between 0.0 (unusable) and 1.0 (excellent).
    """
    scores = [
        _kurtosis_sqi(lead.samples),
        _power_ratio_sqi(lead.samples, lead.sample_rate),
        _peak_regularity_sqi(lead),
        _baseline_sqi(lead.samples, lead.sample_rate),
    ]
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def classify_quality(lead: Lead) -> str:
    """Classify signal quality as a human-readable category.

    Returns
    -------
    str
        ``"excellent"`` (SQI > 0.8), ``"acceptable"`` (0.5--0.8),
        or ``"unacceptable"`` (< 0.5).
    """
    sqi = signal_quality_index(lead)
    if sqi > 0.8:
        return "excellent"
    if sqi >= 0.5:
        return "acceptable"
    return "unacceptable"


def snr_estimate(lead: Lead) -> float:
    """Estimate signal-to-noise ratio in dB.

    Uses a frequency-domain approach: signal power in 1--40 Hz
    vs noise power above 100 Hz (up to Nyquist).

    Parameters
    ----------
    lead : Lead
        Input ECG lead.

    Returns
    -------
    float
        Estimated SNR in decibels.
    """
    sig = require_scipy("signal")

    nperseg = min(256, len(lead.samples))
    freqs, psd = sig.welch(lead.samples, fs=lead.sample_rate, nperseg=nperseg)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    signal_mask = (freqs >= 1.0) & (freqs <= 40.0)
    noise_mask = freqs >= 100.0

    signal_power = float(np.sum(psd[signal_mask]) * df) if np.any(signal_mask) else 0.0
    noise_power = float(np.sum(psd[noise_mask]) * df) if np.any(noise_mask) else 0.0

    if noise_power <= 0:
        return 60.0
    return float(10 * np.log10(signal_power / noise_power))


def _kurtosis_sqi(samples: NDArray[np.float64]) -> float:
    """Kurtosis-based SQI.  ECG signals typically have excess kurtosis 5--15."""
    n = len(samples)
    if n < 4:
        return 0.0
    mean = samples.mean()
    std = samples.std()
    if std == 0:
        return 0.0
    k = float(np.mean(((samples - mean) / std) ** 4)) - 3.0
    if 3.0 <= k <= 20.0:
        return 1.0
    if k < 0:
        return 0.0
    if k < 3.0:
        return k / 3.0
    return max(0.0, 1.0 - (k - 20.0) / 30.0)


def _power_ratio_sqi(samples: NDArray[np.float64], fs: int) -> float:
    """Ratio of power in ECG band (1--40 Hz) to total power."""
    sig = require_scipy("signal")

    nperseg = min(256, len(samples))
    freqs, psd = sig.welch(samples, fs=fs, nperseg=nperseg)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    ecg_mask = (freqs >= 1.0) & (freqs <= 40.0)
    ecg_power = float(np.sum(psd[ecg_mask]) * df) if np.any(ecg_mask) else 0.0
    total_power = float(np.sum(psd) * df)

    if total_power <= 0:
        return 0.0
    return float(np.clip(ecg_power / total_power / 0.7, 0.0, 1.0))


def _peak_regularity_sqi(lead: Lead) -> float:
    """RR-interval regularity: low coefficient of variation means good quality."""
    try:
        peaks = detect_r_peaks(lead)
    except Exception:
        return 0.0
    if len(peaks) < 3:
        return 0.3
    rr = np.diff(peaks).astype(np.float64)
    mean_rr = rr.mean()
    if mean_rr <= 0:
        return 0.0
    cv = rr.std() / mean_rr
    return float(np.clip(1.0 - cv * 2.0, 0.0, 1.0))


def _baseline_sqi(samples: NDArray[np.float64], fs: int) -> float:
    """Baseline stability: low power below 1 Hz indicates stable baseline."""
    sig = require_scipy("signal")

    nperseg = min(256, len(samples))
    freqs, psd = sig.welch(samples, fs=fs, nperseg=nperseg)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    baseline_mask = freqs < 1.0
    ecg_mask = (freqs >= 1.0) & (freqs <= 40.0)

    baseline_power = float(np.sum(psd[baseline_mask]) * df) if np.any(baseline_mask) else 0.0
    ecg_power = float(np.sum(psd[ecg_mask]) * df) if np.any(ecg_mask) else 0.0

    if ecg_power <= 0:
        return 0.0
    return float(np.clip(1.0 - baseline_power / ecg_power, 0.0, 1.0))
