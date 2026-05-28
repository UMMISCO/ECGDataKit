"""ECG signal normalization utilities.

Pure numpy — no scipy required.
"""

from __future__ import annotations

import dataclasses
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import ECGRecord, Lead
from ecgdatakit.processing._core import new_lead


# ---------------------------------------------------------------------------
# normalize_minmax
# ---------------------------------------------------------------------------

@overload
def normalize_minmax(data: Lead) -> Lead: ...
@overload
def normalize_minmax(data: list[Lead]) -> list[Lead]: ...
@overload
def normalize_minmax(data: ECGRecord) -> ECGRecord: ...
@overload
def normalize_minmax(data: list[ECGRecord]) -> list[ECGRecord]: ...
@overload
def normalize_minmax(data: NDArray[np.float64]) -> NDArray[np.float64]: ...

def normalize_minmax(
    data: Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64],
) -> Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64]:
    """Scale signal to the [-1, 1] range.

    Normalization is applied **per lead, per ECG**.

    Accepted inputs:

    * :class:`Lead` — single lead.
    * ``list[Lead]`` — multiple leads (e.g. a 12-lead ECG).
    * :class:`ECGRecord` — all leads and median beats in the record.
    * ``list[ECGRecord]`` — multiple records.
    * 3-D numpy array ``(n_ecgs, n_leads, n_samples)`` — raw multi-ECG data.

    Returns the same type as the input.
    """
    return _dispatch(data, _minmax_samples)


# ---------------------------------------------------------------------------
# normalize_zscore
# ---------------------------------------------------------------------------

@overload
def normalize_zscore(data: Lead) -> Lead: ...
@overload
def normalize_zscore(data: list[Lead]) -> list[Lead]: ...
@overload
def normalize_zscore(data: ECGRecord) -> ECGRecord: ...
@overload
def normalize_zscore(data: list[ECGRecord]) -> list[ECGRecord]: ...
@overload
def normalize_zscore(data: NDArray[np.float64]) -> NDArray[np.float64]: ...

def normalize_zscore(
    data: Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64],
) -> Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64]:
    """Normalize signal to zero mean and unit variance (z-score).

    Normalization is applied **per lead, per ECG**.

    Accepted inputs:

    * :class:`Lead` — single lead.
    * ``list[Lead]`` — multiple leads (e.g. a 12-lead ECG).
    * :class:`ECGRecord` — all leads and median beats in the record.
    * ``list[ECGRecord]`` — multiple records.
    * 3-D numpy array ``(n_ecgs, n_leads, n_samples)`` — raw multi-ECG data.

    Returns the same type as the input.
    """
    return _dispatch(data, _zscore_samples)


# ---------------------------------------------------------------------------
# normalize_amplitude
# ---------------------------------------------------------------------------

@overload
def normalize_amplitude(data: Lead, target_mv: float = 1.0) -> Lead: ...
@overload
def normalize_amplitude(data: list[Lead], target_mv: float = 1.0) -> list[Lead]: ...
@overload
def normalize_amplitude(data: ECGRecord, target_mv: float = 1.0) -> ECGRecord: ...
@overload
def normalize_amplitude(data: list[ECGRecord], target_mv: float = 1.0) -> list[ECGRecord]: ...
@overload
def normalize_amplitude(data: NDArray[np.float64], target_mv: float = 1.0) -> NDArray[np.float64]: ...

def normalize_amplitude(
    data: Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64],
    target_mv: float = 1.0,
) -> Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64]:
    """Scale signal so that its maximum absolute amplitude equals *target_mv*.

    Normalization is applied **per lead, per ECG**.

    Accepted inputs:

    * :class:`Lead` — single lead.
    * ``list[Lead]`` — multiple leads (e.g. a 12-lead ECG).
    * :class:`ECGRecord` — all leads and median beats in the record.
    * ``list[ECGRecord]`` — multiple records.
    * 3-D numpy array ``(n_ecgs, n_leads, n_samples)`` — raw multi-ECG data.

    Returns the same type as the input.

    Parameters
    ----------
    target_mv : float
        Target peak amplitude (default 1.0).
    """
    def _amp(samples: NDArray[np.float64]) -> NDArray[np.float64]:
        return _amplitude_samples(samples, target_mv)

    return _dispatch(data, _amp)


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

from typing import Callable

_SampleFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _normalize_lead(lead: Lead, fn: _SampleFn) -> Lead:
    return new_lead(lead, samples=fn(lead.samples))


def _normalize_record(record: ECGRecord, fn: _SampleFn) -> ECGRecord:
    return dataclasses.replace(
        record,
        leads=[_normalize_lead(ld, fn) for ld in record.leads],
        median_beats=[_normalize_lead(mb, fn) for mb in record.median_beats],
    )


def _normalize_3d(data: NDArray[np.float64], fn: _SampleFn) -> NDArray[np.float64]:
    """Normalize a 3-D array (n_ecgs, n_leads, n_samples) per lead per ECG."""
    result = np.empty_like(data)
    for ecg_idx in range(data.shape[0]):
        for lead_idx in range(data.shape[1]):
            result[ecg_idx, lead_idx] = fn(data[ecg_idx, lead_idx])
    return result


def _dispatch(
    data: Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64],
    fn: _SampleFn,
) -> Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64]:
    if isinstance(data, Lead):
        return _normalize_lead(data, fn)
    if isinstance(data, ECGRecord):
        return _normalize_record(data, fn)
    if isinstance(data, np.ndarray):
        if data.ndim != 3:
            raise ValueError(
                f"numpy input must be 3-D (n_ecgs, n_leads, n_samples), "
                f"got {data.ndim}-D"
            )
        return _normalize_3d(data, fn)
    if isinstance(data, list):
        if not data:
            return []
        first = data[0]
        if isinstance(first, Lead):
            return [_normalize_lead(ld, fn) for ld in data]
        if isinstance(first, ECGRecord):
            return [_normalize_record(rec, fn) for rec in data]
    raise TypeError(
        f"Expected Lead, list[Lead], ECGRecord, list[ECGRecord], "
        f"or 3-D numpy array, got {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# Sample-level normalization functions
# ---------------------------------------------------------------------------

def _minmax_samples(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    vmin, vmax = samples.min(), samples.max()
    span = vmax - vmin
    if span == 0:
        return np.zeros_like(samples)
    return (2.0 * (samples - vmin) / span - 1.0).astype(np.float64)


def _zscore_samples(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    std = samples.std()
    if std == 0:
        return np.zeros_like(samples)
    return ((samples - samples.mean()) / std).astype(np.float64)


def _amplitude_samples(samples: NDArray[np.float64], target_mv: float) -> NDArray[np.float64]:
    peak = np.abs(samples).max()
    if peak == 0:
        return np.zeros_like(samples)
    return (samples * (target_mv / peak)).astype(np.float64)