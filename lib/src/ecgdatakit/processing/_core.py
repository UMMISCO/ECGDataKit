"""Shared infrastructure for the processing subpackage."""

from __future__ import annotations

import dataclasses
import math
import sys
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike


def require_scipy(module: str = "signal") -> ModuleType:
    """Lazily import a scipy submodule, raising a helpful error if missing."""
    try:
        import importlib

        return importlib.import_module(f"scipy.{module}")
    except ImportError as exc:
        raise ImportError(
            f"scipy.{module} is required for this function. "
            'Install it with: pip install "ecgdatakit[processing]"'
        ) from exc


def external_stacklevel() -> int:
    """``stacklevel`` for ``warnings.warn`` that points at the first caller
    outside the ecgdatakit package, however deep the internal call chain."""
    level = 1
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__", "").startswith("ecgdatakit"):
        frame = frame.f_back
        level += 1
    return level


def new_lead(source: Lead, *, samples: NDArray[np.float64], **overrides) -> Lead:
    """Create a new Lead by copying metadata from *source*, replacing samples.

    Parameters
    ----------
    source : Lead
        The lead to copy metadata from (label, sampling_rate, units, etc.).
    samples : NDArray
        New sample data for the returned lead.
    **overrides
        Any Lead field to override (e.g., ``sampling_rate=250``).
    """
    return dataclasses.replace(source, samples=samples, **overrides)


def _check_fs(fs: float) -> None:
    try:
        ok = math.isfinite(fs) and fs > 0
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f"sampling rate must be a positive number, got {fs!r}")


def ensure_lead(
    lead_like: LeadLike, *, fs: int | None = None, label: str = ""
) -> Lead:
    """Coerce a Lead or numpy array into a Lead object.

    Parameters
    ----------
    lead_like : Lead | NDArray[np.float64]
        A Lead object or a 1-D numpy array of samples.  A Lead with float64
        samples is returned as-is; other sample dtypes are converted to
        float64 so later arithmetic cannot overflow.
    fs : int | None
        Sample rate in Hz.  Required when *lead_like* is a numpy array;
        ignored when it is already a Lead.
    label : str
        Lead label to use when constructing from a numpy array (default ``""``).

    Raises
    ------
    TypeError
        If *lead_like* is a numpy array and *fs* is not provided.
    ValueError
        If the samples are not 1-D or the sample rate is not positive.
    """
    if isinstance(lead_like, Lead):
        lead = lead_like
        if lead.samples.dtype != np.float64:
            lead = new_lead(lead, samples=np.asarray(lead.samples, dtype=np.float64))
    else:
        if fs is None:
            raise TypeError(
                "sampling_rate (fs) is required when passing a numpy array "
                "instead of a Lead object"
            )
        lead = Lead(
            label=label,
            samples=np.asarray(lead_like, dtype=np.float64),
            sampling_rate=fs,
        )
    if lead.samples.ndim != 1:
        raise ValueError(
            f"expected a single lead (1-D samples), got shape {lead.samples.shape}; "
            "process multi-lead arrays one row at a time"
        )
    _check_fs(lead.sampling_rate)
    return lead


def require_finite(lead: Lead, func: str) -> None:
    """Raise ``ValueError`` if *lead* has NaN/Inf samples or no samples.

    Recursive filters and resampling spread a single NaN over the whole
    output, so these functions refuse non-finite input instead.
    """
    if lead.samples.size == 0:
        raise ValueError(f"{func}: lead {lead.label!r} has no samples")
    bad = ~np.isfinite(lead.samples)
    if bad.any():
        raise ValueError(
            f"{func}: lead {lead.label!r} has {int(bad.sum())} NaN/Inf samples "
            "(e.g. gaps or padding in the file); fill or trim them first"
        )


def fold_offset(lead: Lead) -> Lead:
    """Fold a raw lead's ``offset`` into its samples.

    For raw leads ``physical = samples * resolution + offset``.  Filters that
    remove DC would otherwise leave the offset to be re-added by
    :meth:`Lead.to_physical`.  Returns *lead* unchanged when there is nothing
    to fold.
    """
    if not lead.is_raw or lead.offset == 0.0 or lead.resolution == 0.0:
        return lead
    return new_lead(
        lead,
        samples=lead.samples + lead.offset / lead.resolution,
        offset=0.0,
    )
