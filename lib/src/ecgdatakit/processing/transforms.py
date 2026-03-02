"""ECG signal transforms and beat segmentation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead, require_scipy
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
        ``"welch"`` (default) for Welch's method.
    nperseg : int | None
        Segment length for Welch's method. Defaults to ``min(256, len(samples))``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(frequencies, power)`` arrays.
    """
    lead = ensure_lead(lead, fs=fs)
    sig = require_scipy("signal")

    if nperseg is None:
        nperseg = min(256, len(lead.samples))

    freqs, psd = sig.welch(
        lead.samples, fs=lead.sample_rate, nperseg=nperseg
    )
    return freqs.astype(np.float64), psd.astype(np.float64)


def fft(lead: LeadLike, *, fs: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the single-sided FFT magnitude spectrum.

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
    n = len(lead.samples)
    yf = np.fft.rfft(lead.samples)
    xf = np.fft.rfftfreq(n, d=1.0 / lead.sample_rate)
    magnitudes = (2.0 / n) * np.abs(yf)
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
        R-peak indices. Detected automatically if ``None``.
    before : float
        Seconds before R-peak to include (default 0.2).
    after : float
        Seconds after R-peak to include (default 0.4).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    list[Lead]
        One Lead per beat, labelled ``"{label}_beat_{i}"``.
    """
    lead = ensure_lead(lead, fs=fs)
    if peaks is None:
        peaks = detect_r_peaks(lead)

    pre = int(round(before * lead.sample_rate))
    post = int(round(after * lead.sample_rate))
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
        R-peak indices. Detected automatically if ``None``.
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
    """
    lead = ensure_lead(lead, fs=fs)
    beats = segment_beats(lead, peaks, before, after)
    if not beats:
        pre = int(round(before * lead.sample_rate))
        post = int(round(after * lead.sample_rate))
        return new_lead(
            lead,
            samples=np.zeros(pre + post, dtype=np.float64),
            label=f"{lead.label}_avg",
        )
    stacked = np.stack([b.samples for b in beats], axis=0)
    avg = stacked.mean(axis=0).astype(np.float64)
    return new_lead(lead, samples=avg, label=f"{lead.label}_avg")
