"""Shared infrastructure for the processing subpackage."""

from __future__ import annotations

import dataclasses
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead


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
