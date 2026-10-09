"""Shared infrastructure for the plotting subpackage."""

from __future__ import annotations

import functools
import math
import warnings
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike


PLOT_STYLE = "seaborn-v0_8-whitegrid"


def require_matplotlib() -> ModuleType:
    """Lazily import matplotlib, raising a helpful error if missing."""
    try:
        import matplotlib
        import matplotlib.pyplot  # noqa: F401

        return matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for static plots. "
            'Install it with: pip install "ecgdatakit[plotting]"'
        ) from exc


def mpl_styled(func):
    """Check matplotlib is installed and draw inside the library style.

    The style is applied with a context manager so the caller's global
    matplotlib settings are left unchanged.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        require_matplotlib()
        import matplotlib.pyplot as plt

        with plt.style.context(PLOT_STYLE):
            return func(*args, **kwargs)

    return wrapper


def require_plotly() -> ModuleType:
    """Lazily import plotly, raising a helpful error if missing."""
    try:
        import plotly

        return plotly
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive plots. "
            'Install it with: pip install "ecgdatakit[plotting-interactive]"'
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
    return _array_lead(lead_like, fs, label)


def _array_lead(values, fs: int, label: str) -> Lead:
    """Wrap a user array: values are taken as given, with no unit."""
    return Lead(
        label=label,
        samples=np.asarray(values, dtype=np.float64),
        sampling_rate=fs,
        is_raw=False,
    )


def check_rate(lead: Lead) -> None:
    """Raise a clear error when a lead has no usable sampling rate."""
    if not lead.sampling_rate or lead.sampling_rate <= 0:
        raise ValueError(
            f"Lead {lead.label!r} has no sampling rate ({lead.sampling_rate!r}), "
            "cannot build a time axis. Use x_axis='samples' or set sampling_rate."
        )


def time_axis(lead: LeadLike, *, fs: int | None = None) -> NDArray[np.float64]:
    """Build a time array in seconds for a lead's samples."""
    lead = ensure_lead(lead, fs=fs)
    check_rate(lead)
    return np.arange(len(lead.samples), dtype=np.float64) / lead.sampling_rate


def amplitude_unit(lead: Lead) -> str:
    """Unit to show for a lead's samples: its unit, ``"raw counts"`` or ``""``."""
    if lead.units:
        return lead.units
    return "raw counts" if lead.is_raw else ""


def amplitude_label(lead: Lead, name: str = "Amplitude") -> str:
    """Axis label with the lead's unit, e.g. ``"Amplitude (mV)"``."""
    unit = amplitude_unit(lead)
    return f"{name} ({unit})" if unit else name


def x_extent(leads: list[Lead], x_axis: str) -> tuple[float, float] | None:
    """Common x range covering every lead, or ``None`` when all are empty."""
    ends = []
    for lead in leads:
        n = len(lead.samples)
        if n == 0:
            continue
        if x_axis == "samples":
            ends.append((1.0, float(n)))
        else:
            check_rate(lead)
            ends.append((0.0, (n - 1) / lead.sampling_rate))
    if not ends:
        return None
    lo = min(e[0] for e in ends)
    hi = max(e[1] for e in ends)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


MAX_POINTS = 200_000
"""Maximum points per interactive trace before min/max decimation."""


def decimate_minmax(
    x: NDArray, y: NDArray, max_points: int = MAX_POINTS,
) -> tuple[NDArray, NDArray, bool]:
    """Reduce a long trace to about *max_points* points, keeping peaks.

    Each bin of samples is replaced by its minimum and maximum (in time
    order), so QRS peaks stay visible. Returns ``(x, y, decimated)``.
    """
    n = len(y)
    if n <= max_points:
        return x, y, False
    size = int(math.ceil(n / (max_points // 2)))
    nb = n // size
    yb = y[: nb * size].reshape(nb, size)
    xb = x[: nb * size].reshape(nb, size)
    i_min = yb.argmin(axis=1)
    i_max = yb.argmax(axis=1)
    first = np.minimum(i_min, i_max)
    second = np.maximum(i_min, i_max)
    rows = np.arange(nb)
    xo = np.empty(2 * nb, dtype=np.float64)
    yo = np.empty(2 * nb, dtype=np.float64)
    xo[0::2], xo[1::2] = xb[rows, first], xb[rows, second]
    yo[0::2], yo[1::2] = yb[rows, first], yb[rows, second]
    if nb * size < n:
        xo = np.append(xo, x[nb * size:])
        yo = np.append(yo, y[nb * size:])
    # Keep the first and last samples so the trace spans the whole lead
    if xo[0] != x[0]:
        xo, yo = np.insert(xo, 0, x[0]), np.insert(yo, 0, y[0])
    if xo[-1] != x[-1]:
        xo, yo = np.append(xo, x[-1]), np.append(yo, y[-1])
    return xo, yo, True


def warn_decimated(labels: list[str], stacklevel: int = 3) -> None:
    """Tell the caller that some traces were decimated.

    The default *stacklevel* points at the caller of the function that calls
    this helper; static plots add one level for the style decorator.
    """
    if labels:
        warnings.warn(
            f"Leads {labels} are longer than {MAX_POINTS} samples and were drawn "
            "with min/max decimation. Slice the lead to see every sample.",
            stacklevel=stacklevel,
        )


def check_rr(rr_ms) -> NDArray[np.float64]:
    """Return RR intervals as a 1-D float array, or raise ValueError."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    if rr.ndim != 1:
        raise ValueError(f"rr_ms must be a 1-D array, got shape {rr.shape}")
    if rr.size == 0:
        raise ValueError("rr_ms is empty")
    if not np.all(np.isfinite(rr)):
        raise ValueError("rr_ms contains NaN or infinite values")
    if np.any(rr <= 0):
        raise ValueError("rr_ms must contain only positive intervals")
    return rr


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
            _array_lead(row, fs, f"Lead {i}") for i, row in enumerate(leads_or_record)
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
            _array_lead(arr, fs, f"Lead {i}") for i, arr in enumerate(leads_or_record)
        ]
        return lead_list, None

    return list(leads_or_record), None


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
