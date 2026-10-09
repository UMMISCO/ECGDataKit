"""ECG signal transforms and beat segmentation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead, require_finite, require_scipy
from ecgdatakit.processing.peaks import detect_r_peaks


def power_spectrum(
    lead: LeadLike,
    method: str = "welch",
    nperseg: int | None = None,
    *,
    fs: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the power spectral density of an ECG lead.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    method : str
        ``"welch"`` (default) for Welch's averaged periodogram, or
        ``"periodogram"`` for a single periodogram of the whole signal.
    nperseg : int | None
        Segment length for Welch's method. Defaults to ``min(256, len(samples))``.
        Not used by ``"periodogram"``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(frequencies, power)`` arrays.
    """
    lead = ensure_lead(lead, fs=fs)
    if method not in ("welch", "periodogram"):
        raise ValueError(f"Unknown method {method!r}; choose 'welch' or 'periodogram'")
    sig = require_scipy("signal")
    require_finite(lead, "power_spectrum")

    if method == "periodogram":
        freqs, psd = sig.periodogram(lead.samples, fs=lead.sampling_rate)
    else:
        if nperseg is None:
            nperseg = min(256, len(lead.samples))
        freqs, psd = sig.welch(lead.samples, fs=lead.sampling_rate, nperseg=nperseg)
    return freqs.astype(np.float64), psd.astype(np.float64)


def fft(lead: LeadLike, *, fs: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the single-sided FFT amplitude spectrum.

    Bins are scaled so a sinusoid of amplitude *A* reads *A*: interior bins
    are doubled, the DC bin (and the Nyquist bin for an even length) are not.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(frequencies, magnitudes)`` arrays (positive frequencies only).
    """
    lead = ensure_lead(lead, fs=fs)
    require_finite(lead, "fft")
    n = len(lead.samples)
    yf = np.fft.rfft(lead.samples)
    xf = np.fft.rfftfreq(n, d=1.0 / lead.sampling_rate)
    magnitudes = (2.0 / n) * np.abs(yf)
    magnitudes[0] /= 2.0
    if n % 2 == 0:
        magnitudes[-1] /= 2.0
    return xf.astype(np.float64), magnitudes.astype(np.float64)


def segment_beats(
    lead: LeadLike,
    peaks: NDArray[np.intp] | None = None,
    before: float = 0.2,
    after: float = 0.4,
    *,
    fs: int | None = None,
) -> list[Lead]:
    """Segment individual heartbeats around R-peaks.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    peaks : NDArray | None
        R-peak sample indices (integers, or whole-number floats).  Detected
        with :func:`detect_r_peaks` and its default method if ``None``.
    before : float
        Seconds before R-peak to include (default 0.2).
    after : float
        Seconds after R-peak to include (default 0.4).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    list[Lead]
        One Lead per beat, labelled ``"{label}_beat_{i}"``.  Beats whose
        window runs past either end of the signal are skipped.
    """
    lead = ensure_lead(lead, fs=fs)
    peaks = detect_r_peaks(lead) if peaks is None else _as_indices(peaks)

    pre = int(round(before * lead.sampling_rate))
    post = int(round(after * lead.sampling_rate))
    n = len(lead.samples)
    beats: list[Lead] = []

    for idx, p in enumerate(peaks):
        lo = p - pre
        hi = p + post
        if lo < 0 or hi > n:
            continue
        segment = lead.samples[lo:hi].copy().astype(np.float64)
        beats.append(
            new_lead(lead, samples=segment, label=f"{lead.label}_beat_{idx}")
        )

    return beats


def _as_indices(peaks) -> NDArray[np.intp]:
    """Return *peaks* as integer indices, refusing non-integral values."""
    arr = np.asarray(peaks)
    if arr.size == 0:
        return np.empty(0, dtype=np.intp)
    if arr.ndim != 1:
        raise ValueError(f"peaks must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind in "iu":
        return arr.astype(np.intp)
    if arr.dtype.kind == "f" and np.isfinite(arr).all() and (arr == np.round(arr)).all():
        return arr.astype(np.intp)
    raise ValueError("peaks must be integer sample indices")


def average_beat(
    lead: LeadLike,
    peaks: NDArray[np.intp] | None = None,
    before: float = 0.2,
    after: float = 0.4,
    *,
    fs: int | None = None,
) -> Lead:
    """Compute the ensemble-averaged heartbeat (template).

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    peaks : NDArray | None
        R-peak sample indices.  Detected with :func:`detect_r_peaks` and its
        default method if ``None``.
    before : float
        Seconds before R-peak (default 0.2).
    after : float
        Seconds after R-peak (default 0.4).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    Lead
        Averaged beat labelled ``"{label}_avg"``.

    Raises
    ------
    ValueError
        If no complete beat is available to average.
    """
    lead = ensure_lead(lead, fs=fs)
    beats = segment_beats(lead, peaks, before, after)
    if not beats:
        raise ValueError(
            f"average_beat: no complete beat in lead {lead.label!r} "
            "(no R-peaks found, or all too close to the signal edges)"
        )
    stacked = np.stack([b.samples for b in beats], axis=0)
    avg = stacked.mean(axis=0).astype(np.float64)
    return new_lead(lead, samples=avg, label=f"{lead.label}_avg")
