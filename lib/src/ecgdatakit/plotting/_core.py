"""Shared infrastructure for the plotting subpackage."""

from __future__ import annotations

from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike


def require_matplotlib() -> ModuleType:
    """Lazily import matplotlib, raising a helpful error if missing."""
    try:
        import matplotlib

        return matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for static plots. "
            "Install it with: pip install ecgdatakit[plotting]"
        ) from exc


def require_plotly() -> ModuleType:
    """Lazily import plotly, raising a helpful error if missing."""
    try:
        import plotly

        return plotly
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install it with: pip install ecgdatakit[plotting-interactive]"
        ) from exc


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


def time_axis(lead: LeadLike, *, fs: int | None = None) -> NDArray[np.float64]:
    """Build a time array in seconds for a lead's samples."""
    lead = ensure_lead(lead, fs=fs)
    return np.arange(len(lead.samples), dtype=np.float64) / lead.sample_rate


GRID_12LEAD: list[list[str]] = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

LEAD_COLORS: dict[str, str] = {
    "I": "#1f77b4",
    "II": "#ff7f0e",
    "III": "#2ca02c",
    "aVR": "#d62728",
    "aVL": "#9467bd",
    "aVF": "#8c564b",
    "V1": "#e377c2",
    "V2": "#7f7f7f",
    "V3": "#bcbd22",
    "V4": "#17becf",
    "V5": "#1a9850",
    "V6": "#d73027",
}

DEFAULT_COLOR = "#333333"

PAPER_SPEED_MM_S = 25
AMPLITUDE_MM_MV = 10


def lead_color(label: str) -> str:
    """Return the colour for a lead label, with a sensible fallback."""
    return LEAD_COLORS.get(label, DEFAULT_COLOR)


def _resolve_leads(leads_or_record):
    """Accept list[Lead] or ECGRecord and return (leads, record|None)."""
    from ecgdatakit.models import ECGRecord

    if isinstance(leads_or_record, ECGRecord):
        return leads_or_record.leads, leads_or_record
    return leads_or_record, None


def _find_lead(leads: list[Lead], label: str) -> Lead | None:
    """Case-insensitive lead lookup."""
    target = label.lower()
    for lead in leads:
        if lead.label.lower() == target:
            return lead
    return None
