"""Shared infrastructure for the processing subpackage."""

from __future__ import annotations

import dataclasses
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
            "Install it with: pip install ecgdatakit[processing]"
        ) from exc


def new_lead(source: Lead, *, samples: NDArray[np.float64], **overrides) -> Lead:
    """Create a new Lead by copying metadata from *source*, replacing samples.

    Parameters
    ----------
    source : Lead
        The lead to copy metadata from (label, sample_rate, units, etc.).
    samples : NDArray
        New sample data for the returned lead.
    **overrides
        Any Lead field to override (e.g., ``sample_rate=250``).
    """
    return dataclasses.replace(source, samples=samples, **overrides)


def ensure_lead(
    lead_like: LeadLike, *, fs: int | None = None, label: str = ""
) -> Lead:
    """Coerce a Lead or numpy array into a Lead object.

    Parameters
    ----------
    lead_like : Lead | NDArray[np.float64]
        A Lead object (returned as-is) or a numpy array of samples.
    fs : int | None
        Sample rate in Hz.  Required when *lead_like* is a numpy array;
        ignored when it is already a Lead.
    label : str
        Lead label to use when constructing from a numpy array (default ``""``).

    Raises
    ------
    TypeError
        If *lead_like* is a numpy array and *fs* is not provided.
    """
    if isinstance(lead_like, Lead):
        return lead_like
    if fs is None:
        raise TypeError(
            "sample_rate (fs) is required when passing a numpy array "
            "instead of a Lead object"
        )
    return Lead(
        label=label,
        samples=np.asarray(lead_like, dtype=np.float64),
        sample_rate=fs,
    )
