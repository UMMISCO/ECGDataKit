"""ECG signal normalization utilities.

Pure numpy — no scipy required.
"""

from __future__ import annotations

from typing import overload

import numpy as np

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead


@overload
def normalize_minmax(lead: LeadLike, *, fs: int | None = None) -> Lead: ...
@overload
def normalize_minmax(lead: list[Lead], *, fs: int | None = None) -> list[Lead]: ...

def normalize_minmax(lead: LeadLike | list[Lead], *, fs: int | None = None) -> Lead | list[Lead]:
    """Scale signal to the [-1, 1] range.

    When *lead* is a list of :class:`Lead`, each lead is normalized
    independently.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64] | list[Lead]
        Input ECG lead, raw signal array, or list of leads.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    if isinstance(lead, list) and (not lead or isinstance(lead[0], Lead)):
        return [normalize_minmax(ld) for ld in lead]
    lead = ensure_lead(lead, fs=fs)
    samples = lead.samples
    vmin, vmax = samples.min(), samples.max()
    span = vmax - vmin
    if span == 0:
        return new_lead(lead, samples=np.zeros_like(samples))
    normalized = 2.0 * (samples - vmin) / span - 1.0
    return new_lead(lead, samples=normalized.astype(np.float64))


@overload
def normalize_zscore(lead: LeadLike, *, fs: int | None = None) -> Lead: ...
@overload
def normalize_zscore(lead: list[Lead], *, fs: int | None = None) -> list[Lead]: ...

def normalize_zscore(lead: LeadLike | list[Lead], *, fs: int | None = None) -> Lead | list[Lead]:
    """Normalize signal to zero mean and unit variance (z-score).

    When *lead* is a list of :class:`Lead`, each lead is normalized
    independently.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64] | list[Lead]
        Input ECG lead, raw signal array, or list of leads.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    if isinstance(lead, list) and (not lead or isinstance(lead[0], Lead)):
        return [normalize_zscore(ld) for ld in lead]
    lead = ensure_lead(lead, fs=fs)
    samples = lead.samples
    std = samples.std()
    if std == 0:
        return new_lead(lead, samples=np.zeros_like(samples))
    normalized = (samples - samples.mean()) / std
    return new_lead(lead, samples=normalized.astype(np.float64))


@overload
def normalize_amplitude(lead: LeadLike, target_mv: float = 1.0, *, fs: int | None = None) -> Lead: ...
@overload
def normalize_amplitude(lead: list[Lead], target_mv: float = 1.0, *, fs: int | None = None) -> list[Lead]: ...

def normalize_amplitude(lead: LeadLike | list[Lead], target_mv: float = 1.0, *, fs: int | None = None) -> Lead | list[Lead]:
    """Scale signal so that its maximum absolute amplitude equals *target_mv*.

    When *lead* is a list of :class:`Lead`, each lead is normalized
    independently.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64] | list[Lead]
        Input ECG lead, raw signal array, or list of leads.
    target_mv : float
        Target peak amplitude (default 1.0).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    if isinstance(lead, list) and (not lead or isinstance(lead[0], Lead)):
        return [normalize_amplitude(ld, target_mv=target_mv) for ld in lead]
    lead = ensure_lead(lead, fs=fs)
    samples = lead.samples
    peak = np.abs(samples).max()
    if peak == 0:
        return new_lead(lead, samples=np.zeros_like(samples))
    normalized = samples * (target_mv / peak)
    return new_lead(lead, samples=normalized.astype(np.float64))
