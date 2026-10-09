"""ECG signal quality assessment.

The signal quality index (SQI) is a heuristic screening score, not a
validated clinical measure.  It averages four sub-scores, each in [0, 1]:

- **Kurtosis** (after Li, Clark and Clifford 2008, kSQI): a clean ECG is
  spiky and has a high excess kurtosis.  Score 1 for excess kurtosis in
  3--20, ``k / 3`` below 3, 0 when negative, and decreasing linearly to 0
  at 50 above 20.
- **Power ratio**: variance of the 1--40 Hz band divided by the variance of
  the mean-removed signal, scaled so a ratio of 0.7 or more scores 1.
- **Peak regularity**: ``1 - 2 * CV`` of the RR intervals from
  :func:`~ecgdatakit.processing.detect_r_peaks`, clipped to [0, 1]; 0.3
  when fewer than 3 peaks are found.  Irregular rhythms (atrial
  fibrillation, ectopy) lower this score although the signal may be clean.
- **Baseline stability**: ``1 - var(< 1 Hz) / var(1--40 Hz)``, clipped to
  [0, 1].

Band components are obtained with zero-phase Butterworth filters, so the
scores do not depend on the sample rate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import LeadLike
from ecgdatakit.processing._core import ensure_lead, require_scipy
from ecgdatakit.processing.peaks import _validate_signal, detect_r_peaks

_SNR_MIN_FS = 200.0


def signal_quality_index(lead: LeadLike, *, fs: int | None = None) -> float:
    """Compute a composite signal quality index (SQI) in the range [0, 1].

    Mean of four sub-scores: kurtosis, power ratio, R-peak regularity and
    baseline stability (see the module documentation for definitions and
    thresholds).

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array (at least 1 s).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    float
        Score between 0.0 (unusable) and 1.0 (excellent).

    Raises
    ------
    ValueError
        Signal shorter than 1 s, not 1-D, or containing NaN or infinite
        values.
    """
    lead = ensure_lead(lead, fs=fs)
    samples = _validate_signal(lead.samples, lead.sampling_rate)
    fs_ = lead.sampling_rate
    scores = [
        _kurtosis_sqi(samples),
        _power_ratio_sqi(samples, fs_),
        _peak_regularity_sqi(samples, fs_),
        _baseline_sqi(samples, fs_),
    ]
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def classify_quality(lead: LeadLike, *, fs: int | None = None) -> str:
    """Classify signal quality as a human-readable category.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    str
        ``"excellent"`` (SQI > 0.8), ``"acceptable"`` (0.5--0.8),
        or ``"unacceptable"`` (< 0.5).
    """
    sqi = signal_quality_index(lead, fs=fs)
    if sqi > 0.8:
        return "excellent"
    if sqi >= 0.5:
        return "acceptable"
    return "unacceptable"


def snr_estimate(lead: LeadLike, *, fs: int | None = None) -> float:
    """Estimate signal-to-noise ratio in dB.

    The noise floor is the mean power spectral density (Welch, 1 s
    segments) above 100 Hz, assumed white and extrapolated over the
    1--40 Hz ECG band.  The SNR is the 1--40 Hz power divided by that
    extrapolated noise power.

    Limits: only broadband noise that reaches above 100 Hz (EMG, electronic
    noise) is measured.  Baseline wander, powerline interference and
    in-band motion artefacts are not counted, and recordings low-pass
    filtered below 100 Hz by the device read as noise-free.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array (at least 1 s).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    float
        Estimated SNR in dB.  ``nan`` when the sample rate is 200 Hz or
        lower (no band above 100 Hz); ``inf`` when no noise is measured.
    """
    lead = ensure_lead(lead, fs=fs)
    samples = _validate_signal(lead.samples, lead.sampling_rate)
    fs_ = lead.sampling_rate
    if fs_ <= _SNR_MIN_FS:
        return float("nan")
    sig = require_scipy("signal")

    nperseg = min(len(samples), int(fs_))
    freqs, psd = sig.welch(samples, fs=fs_, nperseg=nperseg)
    df = freqs[1] - freqs[0]

    signal_mask = (freqs >= 1.0) & (freqs <= 40.0)
    # exclude the half-weight Nyquist bin
    noise_mask = (freqs >= 100.0) & (freqs < fs_ / 2.0)
    if not np.any(noise_mask):
        return float("nan")

    signal_power = float(np.sum(psd[signal_mask]) * df)
    noise_power = float(np.mean(psd[noise_mask])) * 39.0
    if noise_power <= 0:
        return float("inf")
    if signal_power <= 0:
        return float("-inf")
    return float(10 * np.log10(signal_power / noise_power))


def _band(samples: NDArray[np.float64], fs: int, low: float | None, high: float | None) -> NDArray[np.float64]:
    """Zero-phase 2nd-order Butterworth band component."""
    sig = require_scipy("signal")
    if low is not None and high is not None:
        sos = sig.butter(2, [low, high], btype="band", fs=fs, output="sos")
    elif high is not None:
        sos = sig.butter(2, high, btype="low", fs=fs, output="sos")
    else:
        sos = sig.butter(2, low, btype="high", fs=fs, output="sos")
    return sig.sosfiltfilt(sos, samples)


def _ecg_high(fs: int) -> float:
    return min(40.0, 0.45 * fs)


def _kurtosis_sqi(samples: NDArray[np.float64]) -> float:
    """Kurtosis-based SQI.  Clean ECG typically has excess kurtosis 5--15."""
    if len(samples) < 4:
        return 0.0
    std = samples.std()
    if std == 0:
        return 0.0
    k = float(np.mean(((samples - samples.mean()) / std) ** 4)) - 3.0
    if 3.0 <= k <= 20.0:
        return 1.0
    if k < 0:
        return 0.0
    if k < 3.0:
        return k / 3.0
    return max(0.0, 1.0 - (k - 20.0) / 30.0)


def _power_ratio_sqi(samples: NDArray[np.float64], fs: int) -> float:
    """Share of variance in the ECG band (1--40 Hz), 0.7 or more scores 1."""
    total = float(np.var(samples))
    if total <= 0:
        return 0.0
    ecg = float(np.var(_band(samples, fs, 1.0, _ecg_high(fs))))
    return float(np.clip(ecg / total / 0.7, 0.0, 1.0))


def _peak_regularity_sqi(samples: NDArray[np.float64], fs: int) -> float:
    """RR-interval regularity: low coefficient of variation scores high."""
    peaks = detect_r_peaks(samples, fs=fs)
    if len(peaks) < 3:
        return 0.3
    rr = np.diff(peaks).astype(np.float64)
    cv = rr.std() / rr.mean()
    return float(np.clip(1.0 - cv * 2.0, 0.0, 1.0))


def _baseline_sqi(samples: NDArray[np.float64], fs: int) -> float:
    """Baseline stability: low power below 1 Hz relative to 1--40 Hz."""
    centred = samples - samples.mean()
    ecg = float(np.var(_band(centred, fs, 1.0, _ecg_high(fs))))
    if ecg <= 0:
        return 0.0
    baseline = float(np.var(_band(centred, fs, None, 1.0)))
    return float(np.clip(1.0 - baseline / ecg, 0.0, 1.0))
