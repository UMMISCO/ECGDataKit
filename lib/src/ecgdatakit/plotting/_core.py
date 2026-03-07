"""Shared infrastructure for the plotting subpackage."""

from __future__ import annotations

import math
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
            "sampling_rate (fs) is required when passing a numpy array "
            "instead of a Lead object"
        )
    return Lead(
        label=label,
        samples=np.asarray(lead_like, dtype=np.float64),
        sampling_rate=fs,
    )


def time_axis(lead: LeadLike, *, fs: int | None = None) -> NDArray[np.float64]:
    """Build a time array in seconds for a lead's samples."""
    lead = ensure_lead(lead, fs=fs)
    return np.arange(len(lead.samples), dtype=np.float64) / lead.sampling_rate


GRID_12LEAD: list[list[str]] = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

STANDARD_12LEAD: list[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
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


def _resolve_leads(leads_or_record, *, fs: int | None = None):
    """Accept list[Lead], ECGRecord, 2-D ndarray, or list of 1-D ndarrays.

    When numpy arrays are provided, *fs* (sample rate in Hz) is required.
    Returns ``(list[Lead], ECGRecord | None)``.
    """
    from ecgdatakit.models import ECGRecord

    if isinstance(leads_or_record, ECGRecord):
        return leads_or_record.leads, leads_or_record

    # 2-D numpy array  →  one lead per row
    if isinstance(leads_or_record, np.ndarray) and leads_or_record.ndim == 2:
        if fs is None:
            raise TypeError(
                "fs (sample rate) is required when passing a numpy array "
                "instead of Lead objects"
            )
        lead_list = [
            Lead(
                label=f"Lead {i}",
                samples=np.asarray(row, dtype=np.float64),
                sampling_rate=fs,
            )
            for i, row in enumerate(leads_or_record)
        ]
        return lead_list, None

    # list of 1-D numpy arrays
    if (
        isinstance(leads_or_record, list)
        and leads_or_record
        and isinstance(leads_or_record[0], np.ndarray)
    ):
        if fs is None:
            raise TypeError(
                "fs (sample rate) is required when passing numpy arrays "
                "instead of Lead objects"
            )
        lead_list = [
            Lead(
                label=f"Lead {i}",
                samples=np.asarray(arr, dtype=np.float64),
                sampling_rate=fs,
            )
            for i, arr in enumerate(leads_or_record)
        ]
        return lead_list, None

    return leads_or_record, None


def _find_lead(leads: list[Lead], label: str) -> Lead | None:
    """Case-insensitive lead lookup."""
    target = label.lower()
    for lead in leads:
        if lead.label.lower() == target:
            return lead
    return None


def _grid_shape(
    n: int,
    rows: int | None = None,
    cols: int | None = None,
) -> tuple[int, int]:
    """Compute (rows, cols) for a subplot grid holding *n* items.

    If both *rows* and *cols* are given they are returned as-is.
    If only one is given the other is derived.
    If neither is given the default is ``(n, 1)`` (vertical stack).
    """
    if rows and cols:
        return rows, cols
    if rows:
        return rows, math.ceil(n / rows)
    if cols:
        return math.ceil(n / cols), cols
    return n, 1
