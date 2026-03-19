"""ECG signal normalization utilities.

Pure numpy — no scipy required.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead


def _is_lead_list(obj: object) -> bool:
    """Return True if *obj* is a list whose elements are Lead instances."""
    return isinstance(obj, list) and (not obj or isinstance(obj[0], Lead))


def _is_2d_array(obj: object) -> bool:
    """Return True if *obj* is a numpy array with ndim == 2."""
    return isinstance(obj, np.ndarray) and obj.ndim == 2


# ---------------------------------------------------------------------------
# normalize_minmax
# ---------------------------------------------------------------------------

@overload
def normalize_minmax(lead: LeadLike, *, fs: int | None = None) -> Lead: ...
@overload
def normalize_minmax(lead: list[Lead], *, fs: int | None = None) -> list[Lead]: ...
@overload
def normalize_minmax(lead: NDArray[np.float64], *, fs: int | None = None) -> NDArray[np.float64]: ...

def normalize_minmax(lead: LeadLike | list[Lead] | NDArray[np.float64], *, fs: int | None = None) -> Lead | list[Lead] | NDArray[np.float64]:
    """Scale signal to the [-1, 1] range.

    When *lead* is a list of :class:`Lead` or a 2-D numpy array
    (n_leads x n_samples), each lead is normalized independently.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64] | list[Lead]
        Input ECG lead, raw signal array (1-D or 2-D), or list of leads.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a 1-D numpy array.
    """
    if _is_lead_list(lead):
        return [normalize_minmax(ld) for ld in lead]
    if _is_2d_array(lead):
        return np.array([normalize_minmax(row, fs=fs or 1).samples for row in lead])
    lead = ensure_lead(lead, fs=fs)
    samples = lead.samples
    vmin, vmax = samples.min(), samples.max()
    span = vmax - vmin
    if span == 0:
        return new_lead(lead, samples=np.zeros_like(samples))
    normalized = 2.0 * (samples - vmin) / span - 1.0
    return new_lead(lead, samples=normalized.astype(np.float64))


# ---------------------------------------------------------------------------
# normalize_zscore
# ---------------------------------------------------------------------------

@overload
def normalize_zscore(lead: LeadLike, *, fs: int | None = None) -> Lead: ...
@overload
def normalize_zscore(lead: list[Lead], *, fs: int | None = None) -> list[Lead]: ...
@overload
def normalize_zscore(lead: NDArray[np.float64], *, fs: int | None = None) -> NDArray[np.float64]: ...

def normalize_zscore(lead: LeadLike | list[Lead] | NDArray[np.float64], *, fs: int | None = None) -> Lead | list[Lead] | NDArray[np.float64]:
    """Normalize signal to zero mean and unit variance (z-score).

    When *lead* is a list of :class:`Lead` or a 2-D numpy array
    (n_leads x n_samples), each lead is normalized independently.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64] | list[Lead]
        Input ECG lead, raw signal array (1-D or 2-D), or list of leads.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a 1-D numpy array.
    """
    if _is_lead_list(lead):
        return [normalize_zscore(ld) for ld in lead]
    if _is_2d_array(lead):
        return np.array([normalize_zscore(row, fs=fs or 1).samples for row in lead])
    lead = ensure_lead(lead, fs=fs)
    samples = lead.samples
    std = samples.std()
    if std == 0:
        return new_lead(lead, samples=np.zeros_like(samples))
    normalized = (samples - samples.mean()) / std
    return new_lead(lead, samples=normalized.astype(np.float64))


# ---------------------------------------------------------------------------
# normalize_amplitude
# ---------------------------------------------------------------------------

@overload
def normalize_amplitude(lead: LeadLike, target_mv: float = 1.0, *, fs: int | None = None) -> Lead: ...
@overload
def normalize_amplitude(lead: list[Lead], target_mv: float = 1.0, *, fs: int | None = None) -> list[Lead]: ...
@overload
def normalize_amplitude(lead: NDArray[np.float64], target_mv: float = 1.0, *, fs: int | None = None) -> NDArray[np.float64]: ...

def normalize_amplitude(lead: LeadLike | list[Lead] | NDArray[np.float64], target_mv: float = 1.0, *, fs: int | None = None) -> Lead | list[Lead] | NDArray[np.float64]:
    """Scale signal so that its maximum absolute amplitude equals *target_mv*.

    When *lead* is a list of :class:`Lead` or a 2-D numpy array
    (n_leads x n_samples), each lead is normalized independently.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64] | list[Lead]
        Input ECG lead, raw signal array (1-D or 2-D), or list of leads.
    target_mv : float
        Target peak amplitude (default 1.0).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a 1-D numpy array.
    """
    if _is_lead_list(lead):
        return [normalize_amplitude(ld, target_mv=target_mv) for ld in lead]
    if _is_2d_array(lead):
        return np.array([normalize_amplitude(row, target_mv=target_mv, fs=fs or 1).samples for row in lead])
    lead = ensure_lead(lead, fs=fs)
    samples = lead.samples
    peak = np.abs(samples).max()
    if peak == 0:
        return new_lead(lead, samples=np.zeros_like(samples))
    normalized = samples * (target_mv / peak)
    return new_lead(lead, samples=normalized.astype(np.float64))
