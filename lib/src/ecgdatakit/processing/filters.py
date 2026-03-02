"""ECG signal filtering utilities.

All filters use SOS (second-order sections) representation with zero-phase
``sosfiltfilt`` to preserve ECG morphology (no phase distortion).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead, require_scipy


def _validate_nyquist(cutoff: float, fs: int, label: str = "cutoff") -> None:
    nyquist = fs / 2.0
    if cutoff >= nyquist:
        raise ValueError(
            f"{label} ({cutoff} Hz) must be less than Nyquist frequency ({nyquist} Hz)"
        )
    if cutoff <= 0:
        raise ValueError(f"{label} must be positive, got {cutoff}")


def lowpass(lead: LeadLike, cutoff: float, order: int = 4, *, fs: int | None = None) -> Lead:
    """Apply a Butterworth low-pass filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    cutoff : float
        Cutoff frequency in Hz.
    order : int
        Filter order (default 4).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    sig = require_scipy("signal")
    _validate_nyquist(cutoff, lead.sample_rate, "cutoff")
    sos = sig.butter(order, cutoff, btype="low", fs=lead.sample_rate, output="sos")
    filtered = sig.sosfiltfilt(sos, lead.samples).astype(np.float64)
    return new_lead(lead, samples=filtered)


def highpass(lead: LeadLike, cutoff: float, order: int = 4, *, fs: int | None = None) -> Lead:
    """Apply a Butterworth high-pass filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    cutoff : float
        Cutoff frequency in Hz.
    order : int
        Filter order (default 4).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    sig = require_scipy("signal")
    _validate_nyquist(cutoff, lead.sample_rate, "cutoff")
    sos = sig.butter(order, cutoff, btype="high", fs=lead.sample_rate, output="sos")
    filtered = sig.sosfiltfilt(sos, lead.samples).astype(np.float64)
    return new_lead(lead, samples=filtered)


def bandpass(lead: LeadLike, low: float, high: float, order: int = 4, *, fs: int | None = None) -> Lead:
    """Apply a Butterworth band-pass filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    low : float
        Lower cutoff frequency in Hz.
    high : float
        Upper cutoff frequency in Hz.
    order : int
        Filter order (default 4).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    sig = require_scipy("signal")
    _validate_nyquist(high, lead.sample_rate, "high")
    if low <= 0:
        raise ValueError(f"low must be positive, got {low}")
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    sos = sig.butter(order, [low, high], btype="band", fs=lead.sample_rate, output="sos")
    filtered = sig.sosfiltfilt(sos, lead.samples).astype(np.float64)
    return new_lead(lead, samples=filtered)


def notch(lead: LeadLike, freq: float = 50.0, quality: float = 30.0, *, fs: int | None = None) -> Lead:
    """Apply an IIR notch (band-stop) filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    freq : float
        Center frequency to remove (default 50 Hz for mains hum).
    quality : float
        Quality factor — higher means narrower notch (default 30).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    sig = require_scipy("signal")
    _validate_nyquist(freq, lead.sample_rate, "freq")
    b, a = sig.iirnotch(freq, quality, fs=lead.sample_rate)
    sos = sig.tf2sos(b, a)
    filtered = sig.sosfiltfilt(sos, lead.samples).astype(np.float64)
    return new_lead(lead, samples=filtered)


def remove_baseline(lead: LeadLike, cutoff: float = 0.5, order: int = 2, *, fs: int | None = None) -> Lead:
    """Remove baseline wander using a high-pass filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    cutoff : float
        Cutoff frequency in Hz (default 0.5 Hz).
    order : int
        Filter order (default 2).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    return highpass(lead, cutoff=cutoff, order=order)


def diagnostic_filter(lead: LeadLike, notch_freq: float = 50.0, *, fs: int | None = None) -> Lead:
    """Apply AHA diagnostic-grade filtering: 0.05–150 Hz bandpass + notch.

    Suitable for diagnostic ECG interpretation where full morphology
    (including ST segment) must be preserved.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    notch_freq : float
        Power-line frequency to notch out (50 or 60 Hz).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    result = bandpass(lead, low=0.05, high=150.0, order=4)
    result = notch(result, freq=notch_freq)
    return result


def monitoring_filter(lead: LeadLike, notch_freq: float = 50.0, *, fs: int | None = None) -> Lead:
    """Apply monitoring-grade filtering: 0.67–40 Hz bandpass + notch.

    Suitable for arrhythmia monitoring where baseline stability
    is more important than preserving fine morphology.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    notch_freq : float
        Power-line frequency to notch out (50 or 60 Hz).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    result = bandpass(lead, low=0.67, high=40.0, order=4)
    result = notch(result, freq=notch_freq)
    return result
