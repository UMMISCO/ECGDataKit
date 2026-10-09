"""ECG signal normalization utilities.

Pure numpy, no scipy required.

Min-max and z-score outputs are dimensionless: returned leads have
``units=""``, ``resolution=1.0``, ``resolution_unit=""``, ``offset=0.0`` and
``is_raw=False`` so they cannot be mistaken for voltages.  Amplitude
normalization is a pure gain and keeps the lead's units.

NaN samples (gaps in the file) are ignored when computing the statistics
and stay NaN in the output.  A constant signal normalizes to zeros; a
signal whose peak-to-peak range is at most ``1e-12`` times its largest
magnitude counts as constant, so floating-point rounding is not blown up
to +-1.  A flat line around zero (e.g. a high-passed constant, which is
pure rounding noise near 0) has no scale to compare with and is
normalized like any other signal.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, overload

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import _TO_UV, ECGRecord, Lead, _normalize_unit
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
    * numpy array: 1-D ``(n_samples,)``, 2-D ``(n_leads, n_samples)`` or
      3-D ``(n_ecgs, n_leads, n_samples)``; each row along the last axis is
      normalized independently and the result is float64.

    Returns the same type as the input.
    """
    return _dispatch(data, _minmax_samples, _dimensionless(_minmax_samples))


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
    * numpy array: 1-D ``(n_samples,)``, 2-D ``(n_leads, n_samples)`` or
      3-D ``(n_ecgs, n_leads, n_samples)``; each row along the last axis is
      normalized independently and the result is float64.

    Returns the same type as the input.
    """
    return _dispatch(data, _zscore_samples, _dimensionless(_zscore_samples))


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
    * numpy array: 1-D ``(n_samples,)``, 2-D ``(n_leads, n_samples)`` or
      3-D ``(n_ecgs, n_leads, n_samples)``; each row along the last axis is
      normalized independently and the result is float64.

    Returns the same type as the input.

    The scaling is a pure gain on the physical signal, so the lead keeps
    its units and scaling metadata.  For a lead in a known voltage unit
    (``units`` for physical leads, ``resolution_unit`` for raw leads) the
    peak of the physical signal becomes *target_mv* millivolts, expressed
    in that unit (e.g. 1000 for a ``"uV"`` lead).  Without a known unit, and
    for numpy input, the peak becomes *target_mv* in the signal's own units.

    Parameters
    ----------
    target_mv : float
        Target peak amplitude in mV (default 1.0).
    """
    def _amp(samples: NDArray[np.float64]) -> NDArray[np.float64]:
        return _amplitude_samples(samples, target_mv)

    def _amp_lead(lead: Lead) -> Lead:
        return _amplitude_lead(lead, target_mv)

    return _dispatch(data, _amp, _amp_lead)


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

_SampleFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]
_LeadFn = Callable[[Lead], Lead]


def _dimensionless(fn: _SampleFn) -> _LeadFn:
    """Wrap *fn* so the returned lead carries no physical unit or scaling."""

    def apply(lead: Lead) -> Lead:
        samples = np.asarray(lead.samples, dtype=np.float64)
        return new_lead(
            lead,
            samples=fn(samples),
            units="",
            resolution=1.0,
            resolution_unit="",
            offset=0.0,
            is_raw=False,
        )

    return apply


def _amplitude_lead(lead: Lead, target_mv: float) -> Lead:
    samples = np.asarray(lead.samples, dtype=np.float64)
    unit = _normalize_unit(lead.resolution_unit if lead.is_raw else lead.units)
    target = target_mv * _TO_UV["mV"] / _TO_UV[unit] if unit else target_mv
    if lead.is_raw and lead.resolution != 0.0:
        physical = samples * lead.resolution + lead.offset
    else:
        physical = samples
    peak = _nan_peak(physical)
    if peak is None:
        return new_lead(lead, samples=samples.copy())
    if peak == 0:
        return new_lead(lead, samples=np.where(np.isnan(samples), np.nan, 0.0), offset=0.0)
    gain = target / peak
    return new_lead(lead, samples=samples * gain, offset=lead.offset * gain)


def _normalize_record(record: ECGRecord, fn: _LeadFn) -> ECGRecord:
    return dataclasses.replace(
        record,
        leads=[fn(ld) for ld in record.leads],
        median_beats=[fn(mb) for mb in record.median_beats],
    )


def _normalize_array(data: NDArray, fn: _SampleFn) -> NDArray[np.float64]:
    """Normalize each row along the last axis of a 1-D, 2-D or 3-D array."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return fn(data)
    result = np.empty(data.shape, dtype=np.float64)
    for idx in np.ndindex(data.shape[:-1]):
        result[idx] = fn(data[idx])
    return result


def _dispatch(
    data: Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64],
    fn: _SampleFn,
    lead_fn: _LeadFn,
) -> Lead | list[Lead] | ECGRecord | list[ECGRecord] | NDArray[np.float64]:
    if isinstance(data, Lead):
        return lead_fn(data)
    if isinstance(data, ECGRecord):
        return _normalize_record(data, lead_fn)
    if isinstance(data, np.ndarray):
        if data.ndim not in (1, 2, 3):
            raise ValueError(
                "numpy input must be 1-D (n_samples,), 2-D (n_leads, n_samples) "
                f"or 3-D (n_ecgs, n_leads, n_samples), got {data.ndim}-D"
            )
        return _normalize_array(data, fn)
    if isinstance(data, list):
        if not data:
            return []
        if all(isinstance(ld, Lead) for ld in data):
            return [lead_fn(ld) for ld in data]
        if all(isinstance(rec, ECGRecord) for rec in data):
            return [_normalize_record(rec, lead_fn) for rec in data]
    raise TypeError(
        f"Expected Lead, list[Lead], ECGRecord, list[ECGRecord], "
        f"or a 1-D/2-D/3-D numpy array, got {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# Sample-level normalization functions (NaN-aware)
# ---------------------------------------------------------------------------

def _nan_peak(samples: NDArray[np.float64]) -> float | None:
    """Largest absolute value ignoring NaN, ``None`` if there is none."""
    if samples.size == 0 or np.isnan(samples).all():
        return None
    return float(np.nanmax(np.abs(samples)))


_CONSTANT_RTOL = 1e-12
"""Peak-to-peak below this fraction of the peak magnitude counts as constant."""


def _constant_or_empty(samples: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Result for empty, all-NaN or constant input, else ``None``."""
    if samples.size == 0 or np.isnan(samples).all():
        return samples.astype(np.float64, copy=True)
    vmax, vmin = np.nanmax(samples), np.nanmin(samples)
    # Constant up to rounding (e.g. [0.1 + 0.2, 0.3]): scaling the last-bit
    # noise up to +-1 would invent a signal
    if vmax - vmin <= _CONSTANT_RTOL * max(abs(vmax), abs(vmin)):
        return np.where(np.isnan(samples), np.nan, 0.0)
    return None


def _minmax_samples(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    special = _constant_or_empty(samples)
    if special is not None:
        return special
    vmin, vmax = np.nanmin(samples), np.nanmax(samples)
    return (2.0 * (samples - vmin) / (vmax - vmin) - 1.0).astype(np.float64)


def _zscore_samples(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    special = _constant_or_empty(samples)
    if special is not None:
        return special
    return ((samples - np.nanmean(samples)) / np.nanstd(samples)).astype(np.float64)


def _amplitude_samples(samples: NDArray[np.float64], target_mv: float) -> NDArray[np.float64]:
    peak = _nan_peak(samples)
    if peak is None:
        return samples.astype(np.float64, copy=True)
    if peak == 0:
        return np.where(np.isnan(samples), np.nan, 0.0)
    return (samples * (target_mv / peak)).astype(np.float64)
