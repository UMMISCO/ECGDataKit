"""ECG signal filtering utilities.

All filters use SOS (second-order sections) representation with zero-phase
``sosfiltfilt`` to preserve ECG morphology (no phase distortion).

Input must be finite: a NaN or Inf sample would spread over the whole
filtered output, so the filters raise ``ValueError`` instead.  Raw leads
are filtered in ADC counts; their ``offset`` is folded into the samples so
that :meth:`~ecgdatakit.models.Lead.to_physical` stays correct afterwards.
"""

from __future__ import annotations

import math
import operator
import warnings

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import (
    ensure_lead,
    external_stacklevel,
    fold_offset,
    new_lead,
    require_finite,
    require_scipy,
)


def _validate_cutoff(cutoff: float, fs: float, label: str = "cutoff") -> None:
    nyquist = fs / 2.0
    if not isinstance(cutoff, (int, float, np.number)) or not math.isfinite(cutoff):
        raise ValueError(f"{label} must be a finite number, got {cutoff!r}")
    if cutoff <= 0:
        raise ValueError(f"{label} must be positive, got {cutoff}")
    if cutoff >= nyquist:
        raise ValueError(
            f"{label} ({cutoff} Hz) must be less than Nyquist frequency ({nyquist} Hz)"
        )


def _validate_order(order: int) -> int:
    try:
        order = operator.index(order)
    except TypeError:
        raise ValueError(f"order must be a positive integer, got {order!r}") from None
    if order < 1:
        raise ValueError(f"order must be a positive integer, got {order}")
    return order


def _apply_sos(lead: Lead, sos: NDArray[np.float64], func: str) -> Lead:
    """Zero-phase filter *lead* with *sos* after checking its length."""
    sig = require_scipy("signal")
    require_finite(lead, func)
    # Same default padding as scipy.signal.sosfiltfilt
    ntaps = 2 * len(sos) + 1
    ntaps -= min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    padlen = 3 * ntaps
    n = len(lead.samples)
    if n <= padlen:
        raise ValueError(
            f"{func}: signal too short ({n} samples); this filter needs more "
            f"than {padlen} samples"
        )
    lead = fold_offset(lead)
    filtered = sig.sosfiltfilt(sos, lead.samples).astype(np.float64)
    return new_lead(lead, samples=filtered)


def lowpass(lead: LeadLike, cutoff: float, order: int = 4, *, fs: int | None = None) -> Lead:
    """Apply a Butterworth low-pass filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or 1-D signal array.
    cutoff : float
        Cutoff frequency in Hz.
    order : int
        Filter order (default 4).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    sig = require_scipy("signal")
    order = _validate_order(order)
    _validate_cutoff(cutoff, lead.sampling_rate, "cutoff")
    sos = sig.butter(order, cutoff, btype="low", fs=lead.sampling_rate, output="sos")
    return _apply_sos(lead, sos, "lowpass")


def highpass(lead: LeadLike, cutoff: float, order: int = 4, *, fs: int | None = None) -> Lead:
    """Apply a Butterworth high-pass filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or 1-D signal array.
    cutoff : float
        Cutoff frequency in Hz.
    order : int
        Filter order (default 4).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    sig = require_scipy("signal")
    order = _validate_order(order)
    _validate_cutoff(cutoff, lead.sampling_rate, "cutoff")
    sos = sig.butter(order, cutoff, btype="high", fs=lead.sampling_rate, output="sos")
    return _apply_sos(lead, sos, "highpass")


def bandpass(lead: LeadLike, low: float, high: float, order: int = 4, *, fs: int | None = None) -> Lead:
    """Apply a Butterworth band-pass filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or 1-D signal array.
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
    order = _validate_order(order)
    _validate_cutoff(low, lead.sampling_rate, "low")
    _validate_cutoff(high, lead.sampling_rate, "high")
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    sos = sig.butter(order, [low, high], btype="band", fs=lead.sampling_rate, output="sos")
    return _apply_sos(lead, sos, "bandpass")


def notch(lead: LeadLike, freq: float = 50.0, quality: float = 30.0, *, fs: int | None = None) -> Lead:
    """Apply an IIR notch (band-stop) filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or 1-D signal array.
    freq : float
        Center frequency to remove (default 50 Hz for mains hum).
    quality : float
        Quality factor, higher means narrower notch (default 30).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    sig = require_scipy("signal")
    _validate_cutoff(freq, lead.sampling_rate, "freq")
    if not isinstance(quality, (int, float, np.number)) or not (
        math.isfinite(quality) and quality > 0
    ):
        raise ValueError(f"quality must be a positive number, got {quality!r}")
    b, a = sig.iirnotch(freq, quality, fs=lead.sampling_rate)
    sos = sig.tf2sos(b, a)
    return _apply_sos(lead, sos, "notch")


def _notch_if_possible(lead: Lead, freq: float | None, func: str) -> Lead:
    """Apply the power-line notch, skipping it (with a warning) above Nyquist."""
    if freq is None:
        return lead
    if freq >= lead.sampling_rate / 2.0:
        warnings.warn(
            f"{func}: power-line notch at {freq} Hz skipped, it is not below "
            f"the Nyquist frequency ({lead.sampling_rate / 2.0} Hz)",
            stacklevel=external_stacklevel(),
        )
        return lead
    return notch(lead, freq=freq)


def _band_top(high: float, fs: float, func: str) -> float:
    """Clamp a preset's upper cutoff to 0.45 * fs, with a warning."""
    limit = 0.45 * fs
    if high < limit:
        return high
    warnings.warn(
        f"{func}: upper cutoff lowered from {high} Hz to {limit:g} Hz "
        f"because the sampling rate is {fs} Hz",
        stacklevel=external_stacklevel(),
    )
    return limit


def remove_baseline(lead: LeadLike, cutoff: float = 0.5, order: int = 2, *, fs: int | None = None) -> Lead:
    """Remove baseline wander using a high-pass filter.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or 1-D signal array.
    cutoff : float
        Cutoff frequency in Hz (default 0.5 Hz).
    order : int
        Filter order (default 2).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    return highpass(lead, cutoff=cutoff, order=order)


def diagnostic_filter(lead: LeadLike, notch_freq: float | None = 50.0, *, fs: int | None = None) -> Lead:
    """Apply AHA diagnostic-grade filtering: 0.05–150 Hz bandpass + notch.

    Suitable for diagnostic ECG interpretation where full morphology
    (including ST segment) must be preserved.

    The 150 Hz upper cutoff needs a sampling rate above 333 Hz.  At lower
    rates (e.g. 180 or 250 Hz Holter recordings) it is lowered to
    ``0.45 * fs`` and a warning is emitted.  The notch is skipped with a
    warning when *notch_freq* is not below the Nyquist frequency.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or 1-D signal array.
    notch_freq : float | None
        Power-line frequency to notch out (50 or 60 Hz), ``None`` for no notch.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    high = _band_top(150.0, lead.sampling_rate, "diagnostic_filter")
    result = bandpass(lead, low=0.05, high=high, order=4)
    return _notch_if_possible(result, notch_freq, "diagnostic_filter")


def monitoring_filter(lead: LeadLike, notch_freq: float | None = 50.0, *, fs: int | None = None) -> Lead:
    """Apply monitoring-grade filtering: 0.67–40 Hz bandpass + notch.

    Suitable for arrhythmia monitoring where baseline stability
    is more important than preserving fine morphology.  As in
    :func:`diagnostic_filter`, the upper cutoff is lowered to ``0.45 * fs``
    at low sampling rates and an impossible notch is skipped, both with a
    warning.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or 1-D signal array.
    notch_freq : float | None
        Power-line frequency to notch out (50 or 60 Hz), ``None`` for no notch.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    high = _band_top(40.0, lead.sampling_rate, "monitoring_filter")
    result = bandpass(lead, low=0.67, high=high, order=4)
    return _notch_if_possible(result, notch_freq, "monitoring_filter")
