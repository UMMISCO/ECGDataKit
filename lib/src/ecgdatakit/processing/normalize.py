"""ECG signal normalization utilities.

Pure numpy — no scipy required.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead
from ecgdatakit.processing._core import ensure_lead, new_lead


def _is_lead_list(obj: object) -> bool:
    """Return True if *obj* is a list whose elements are Lead instances."""
    return isinstance(obj, list) and (not obj or isinstance(obj[0], Lead))


# ---------------------------------------------------------------------------
# normalize_minmax
# ---------------------------------------------------------------------------

@overload
def normalize_minmax(lead: Lead, *, fs: int | None = None) -> Lead: ...
@overload
def normalize_minmax(lead: NDArray[np.float64], *, fs: int | None = None) -> NDArray[np.float64]: ...
@overload
def normalize_minmax(lead: list[Lead], *, fs: int | None = None) -> list[Lead]: ...

def normalize_minmax(lead: Lead | NDArray[np.float64] | list[Lead], *, fs: int | None = None) -> Lead | NDArray[np.float64] | list[Lead]:
    """Scale signal to the [-1, 1] range.

    When *lead* is a list of :class:`Lead`, a 2-D numpy array
    (n_leads x n_samples), or a 3-D array (n_ecgs x n_leads x n_samples),
    each lead is normalized independently.

    Returns the same type as the input: :class:`Lead` in → :class:`Lead`
    out, numpy array in → numpy array out.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64] | list[Lead]
        Input ECG lead, raw signal array (1-D, 2-D, or 3-D), or list of leads.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a 1-D numpy array.
    """
    if _is_lead_list(lead):
        return [normalize_minmax(ld) for ld in lead]
    if isinstance(lead, np.ndarray):
        if lead.ndim == 1:
            return _minmax_samples(lead)
        shape = lead.shape
        flat = lead.reshape(-1, shape[-1])
        return np.array([_minmax_samples(row) for row in flat]).reshape(shape)
    lead = ensure_lead(lead, fs=fs)
    return new_lead(lead, samples=_minmax_samples(lead.samples))


def _minmax_samples(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    vmin, vmax = samples.min(), samples.max()
    span = vmax - vmin
    if span == 0:
        return np.zeros_like(samples)
    return (2.0 * (samples - vmin) / span - 1.0).astype(np.float64)


# ---------------------------------------------------------------------------
# normalize_zscore
# ---------------------------------------------------------------------------

@overload
def normalize_zscore(lead: Lead, *, fs: int | None = None) -> Lead: ...
@overload
def normalize_zscore(lead: NDArray[np.float64], *, fs: int | None = None) -> NDArray[np.float64]: ...
@overload
def normalize_zscore(lead: list[Lead], *, fs: int | None = None) -> list[Lead]: ...

def normalize_zscore(lead: Lead | NDArray[np.float64] | list[Lead], *, fs: int | None = None) -> Lead | NDArray[np.float64] | list[Lead]:
    """Normalize signal to zero mean and unit variance (z-score).

    When *lead* is a list of :class:`Lead`, a 2-D numpy array
    (n_leads x n_samples), or a 3-D array (n_ecgs x n_leads x n_samples),
    each lead is normalized independently.

    Returns the same type as the input: :class:`Lead` in → :class:`Lead`
    out, numpy array in → numpy array out.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64] | list[Lead]
        Input ECG lead, raw signal array (1-D, 2-D, or 3-D), or list of leads.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a 1-D numpy array.
    """
    if _is_lead_list(lead):
        return [normalize_zscore(ld) for ld in lead]
    if isinstance(lead, np.ndarray):
        if lead.ndim == 1:
            return _zscore_samples(lead)
        shape = lead.shape
        flat = lead.reshape(-1, shape[-1])
        return np.array([_zscore_samples(row) for row in flat]).reshape(shape)
    lead = ensure_lead(lead, fs=fs)
    return new_lead(lead, samples=_zscore_samples(lead.samples))


def _zscore_samples(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    std = samples.std()
    if std == 0:
        return np.zeros_like(samples)
    return ((samples - samples.mean()) / std).astype(np.float64)


# ---------------------------------------------------------------------------
# normalize_amplitude
# ---------------------------------------------------------------------------

@overload
def normalize_amplitude(lead: Lead, target_mv: float = 1.0, *, fs: int | None = None) -> Lead: ...
@overload
def normalize_amplitude(lead: NDArray[np.float64], target_mv: float = 1.0, *, fs: int | None = None) -> NDArray[np.float64]: ...
@overload
def normalize_amplitude(lead: list[Lead], target_mv: float = 1.0, *, fs: int | None = None) -> list[Lead]: ...

def normalize_amplitude(lead: Lead | NDArray[np.float64] | list[Lead], target_mv: float = 1.0, *, fs: int | None = None) -> Lead | NDArray[np.float64] | list[Lead]:
    """Scale signal so that its maximum absolute amplitude equals *target_mv*.

    When *lead* is a list of :class:`Lead`, a 2-D numpy array
    (n_leads x n_samples), or a 3-D array (n_ecgs x n_leads x n_samples),
    each lead is normalized independently.

    Returns the same type as the input: :class:`Lead` in → :class:`Lead`
    out, numpy array in → numpy array out.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64] | list[Lead]
        Input ECG lead, raw signal array (1-D, 2-D, or 3-D), or list of leads.
    target_mv : float
        Target peak amplitude (default 1.0).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a 1-D numpy array.
    """
    if _is_lead_list(lead):
        return [normalize_amplitude(ld, target_mv=target_mv) for ld in lead]
    if isinstance(lead, np.ndarray):
        if lead.ndim == 1:
            return _amplitude_samples(lead, target_mv)
        shape = lead.shape
        flat = lead.reshape(-1, shape[-1])
        return np.array([_amplitude_samples(row, target_mv) for row in flat]).reshape(shape)
    lead = ensure_lead(lead, fs=fs)
    return new_lead(lead, samples=_amplitude_samples(lead.samples, target_mv))


def _amplitude_samples(samples: NDArray[np.float64], target_mv: float) -> NDArray[np.float64]:
    peak = np.abs(samples).max()
    if peak == 0:
        return np.zeros_like(samples)
    return (samples * (target_mv / peak)).astype(np.float64)
