"""Static ECG plots using matplotlib.

All public functions return ``matplotlib.figure.Figure``.
Functions that accept an *ax* parameter can render into an existing axes
for composability; when *ax* is ``None`` a new figure is created.

Requires: ``pip install "ecgdatakit[plotting]"``
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import ECGRecord, Lead, LeadLike
from ecgdatakit.plotting._core import (
    GRID_12LEAD,
    STANDARD_12LEAD,
    _find_lead,
    _grid_shape,
    _resolve_leads,
    amplitude_label,
    amplitude_unit,
    check_rate,
    check_rr,
    decimate_minmax,
    ensure_lead,
    lead_color,
    mpl_styled,
    require_matplotlib,
    time_axis,
    warn_decimated,
    x_extent,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure



def _get_or_create_ax(figsize, ax):
    """Return (fig, ax).  Creates new ones when *ax* is ``None``."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


# Amplitude of one 10 mm paper square (0.5 mV) per voltage unit
_HALF_MV = {"mV": 0.5, "uV": 500.0, "V": 0.0005}
_MAX_TICKS = 500


def _ecg_grid(ax, unit: str = "mV", time_x: bool = True) -> None:
    """Draw ECG paper-style grid lines on *ax*.

    Time lines follow the paper at 0.2 s (major) and 0.04 s (minor), with
    labels on whole seconds only. Amplitude lines are 0.5 mV and 0.1 mV,
    converted to the lead *unit*; for raw counts or an unknown unit the
    amplitude axis keeps matplotlib's automatic ticks. Call after the axis
    limits are set. An axis whose range would need more than 500 lines
    falls back to automatic ticks.
    """
    from matplotlib.ticker import AutoLocator, AutoMinorLocator, FuncFormatter, MultipleLocator

    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#ffcccc", linewidth=0.8)
    ax.grid(True, which="minor", color="#ffe6e6", linewidth=0.4)

    x_lo, x_hi = ax.get_xlim()
    if time_x and (x_hi - x_lo) / 0.04 < _MAX_TICKS:
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{v:.0f}" if abs(v - round(v)) < 1e-6 else ""
        ))
    else:
        ax.xaxis.set_major_locator(AutoLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    half = _HALF_MV.get(unit)
    y_lo, y_hi = ax.get_ylim()
    if half is not None and (y_hi - y_lo) / (half / 5) < _MAX_TICKS:
        ax.yaxis.set_major_locator(MultipleLocator(half))
        ax.yaxis.set_minor_locator(MultipleLocator(half / 5))
    else:
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def _style_ax(ax) -> None:
    """Remove top/right spines and use readable x ticks."""
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator())


def _finish_ax(ax, extent, show_grid: bool, unit: str, x_axis: str) -> None:
    """Set the x range, style the axes and draw the paper grid if asked."""
    if extent is not None:
        ax.set_xlim(*extent)
    _style_ax(ax)
    if show_grid:
        _ecg_grid(ax, unit, time_x=(x_axis != "samples"))


# Static plots add the style decorator frame between the plot and the caller
_STATIC_STACKLEVEL = 4


def _plot_trace(ax, lead, x, decimated: list[str], **kwargs) -> None:
    """Plot a lead, with min/max decimation when it is very long."""
    xs, ys, cut = decimate_minmax(x, lead.samples)
    if cut:
        decimated.append(lead.label)
    ax.plot(xs, ys, **kwargs)


def _x_data(lead, x_axis):
    """Return ``(x_array, xlabel)`` based on *x_axis* mode."""
    if x_axis == "samples":
        return np.arange(1, len(lead.samples) + 1), "Sample"
    return time_axis(lead), "Time (s)"


@mpl_styled
def plot_lead(
    lead: LeadLike,
    peaks: NDArray[np.intp] | None = None,
    title: str | None = None,
    show_grid: bool = False,
    figsize: tuple[float, float] = (12, 3),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
    show: bool = True,
    x_axis: str = "time",
) -> Figure:
    """Plot a single ECG lead waveform.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        ECG lead or raw signal array to plot.
    peaks : NDArray | None
        Optional R-peak indices to mark.
    title : str | None
        Figure title. Defaults to the lead label.
    show_grid : bool
        Draw ECG paper-style grid (default ``False``).
    figsize : tuple
        Figure size in inches (default ``(12, 3)``).
    ax : Axes | None
        Existing axes to draw on. A new figure is created if ``None``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    show : bool
        Display the plot immediately (default ``True``). Set to ``False``
        to return the figure without displaying.
    x_axis : str
        ``"time"`` for seconds (default) or ``"samples"`` for sample indices
        (1, 2, ..., N).
    """
    lead = ensure_lead(lead, fs=fs)
    own_fig = ax is None
    fig, ax = _get_or_create_ax(figsize, ax)
    x, xlabel = _x_data(lead, x_axis)

    decimated: list[str] = []
    _plot_trace(ax, lead, x, decimated, color=lead_color(lead.label), linewidth=0.8)
    warn_decimated(decimated, _STATIC_STACKLEVEL)

    if peaks is not None and len(peaks) > 0:
        ax.plot(
            x[peaks],
            lead.samples[peaks],
            "rv",
            markersize=6,
            label="R-peaks",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(amplitude_label(lead))
    ax.set_title(title or lead.label)

    _finish_ax(ax, x_extent([lead], x_axis), show_grid, amplitude_unit(lead), x_axis)
    fig.tight_layout()

    if show and own_fig:
        import matplotlib.pyplot as plt
        plt.show()

    return fig


@mpl_styled
def plot_leads(
    leads: list[Lead] | ECGRecord | NDArray[np.float64] | list[NDArray[np.float64]],
    peaks_dict: dict[str, NDArray[np.intp]] | None = None,
    title: str | None = None,
    show_grid: bool = False,
    figsize: tuple[float, float | None] = (12, None),
    share_x: bool = True,
    *,
    fs: int | None = None,
    show: bool = True,
    x_axis: str = "time",
    rows: int | None = None,
    cols: int | None = None,
) -> Figure:
    """Plot multiple leads in a grid layout (vertical stack by default).

    Parameters
    ----------
    leads : list[Lead] | ECGRecord | NDArray | list[NDArray]
        Leads to plot.  Also accepts a 2-D numpy array
        (n_leads × n_samples) or a list of 1-D numpy arrays.
    peaks_dict : dict | None
        ``{label: peaks_array}`` for per-lead R-peak markers.
    title : str | None
        Overall figure title.
    show_grid : bool
        Draw ECG paper-style grid.
    figsize : tuple
        Width is fixed; height is auto-calculated (2 in per row) when ``None``.
    share_x : bool
        Share the x-axis across all subplots (default ``True``).
    fs : int | None
        Sample rate in Hz.  Required when *leads* is a numpy array.
    show : bool
        Display the plot immediately (default ``True``).
    x_axis : str
        ``"time"`` for seconds (default) or ``"samples"`` for sample indices.
    rows : int | None
        Number of rows in the subplot grid. Derived from *cols* or defaults
        to one row per lead when neither is given.
    cols : int | None
        Number of columns in the subplot grid (default ``1``).
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    lead_list, _ = _resolve_leads(leads, fs=fs)
    n = len(lead_list)
    if n == 0:
        fig, _ = plt.subplots(figsize=(figsize[0], 3))
        return fig

    r, c = _grid_shape(n, rows, cols)
    h = figsize[1] if figsize[1] is not None else max(3, 2 * r)
    fig, axes = plt.subplots(r, c, figsize=(figsize[0], h), sharex=share_x, squeeze=False)
    # Shared axes span the longest lead so no lead is cut off
    common = x_extent(lead_list, x_axis) if share_x else None

    decimated: list[str] = []
    for i, ld in enumerate(lead_list):
        ri, ci = divmod(i, c)
        ax = axes[ri][ci]
        x, _ = _x_data(ld, x_axis)
        _plot_trace(ax, ld, x, decimated, color=lead_color(ld.label), linewidth=0.8)

        if peaks_dict and ld.label in peaks_dict:
            pk = peaks_dict[ld.label]
            ax.plot(x[pk], ld.samples[pk], "rv", markersize=5)

        unit = amplitude_unit(ld)
        ax.set_ylabel(f"{ld.label}\n({unit})" if unit else ld.label,
                      rotation=0, labelpad=30, fontsize=10)
        ax.yaxis.set_label_position("left")
        _finish_ax(ax, common or x_extent([ld], x_axis), show_grid, unit, x_axis)
    warn_decimated(decimated, _STATIC_STACKLEVEL)

    # Hide empty subplots
    for j in range(n, r * c):
        ri, ci = divmod(j, c)
        axes[ri][ci].set_visible(False)

    # X-axis label on bottom row
    for ci in range(c):
        axes[-1][ci].set_xlabel("Sample" if x_axis == "samples" else "Time (s)")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


@mpl_styled
def plot_12lead(
    leads: list[Lead] | ECGRecord | NDArray[np.float64] | list[NDArray[np.float64]],
    record: ECGRecord | None = None,
    title: str | None = None,
    show_grid: bool = False,
    figsize: tuple[float, float | None] = (12, None),
    share_x: bool = True,
    *,
    fs: int | None = None,
    show: bool = True,
    x_axis: str = "time",
    rows: int | None = None,
    cols: int | None = None,
) -> Figure:
    """Plot 12 leads with standard lead names (I, II, III, aVR, …, V6).

    Unlike :func:`plot_leads`, this function assigns the standard 12-lead
    names (in order) when the input is a numpy array.
    The full signal is plotted without cropping; with a shared x-axis the
    range covers the longest lead.

    Parameters
    ----------
    leads : list[Lead] | ECGRecord | NDArray | list[NDArray]
        Leads (or full record) to plot.  Also accepts a 2-D numpy array
        (n_leads × n_samples) or a list of 1-D numpy arrays.
    record : ECGRecord | None
        If provided, a header with patient/device/measurement info is shown.
    title : str | None
        Overall figure title.
    show_grid : bool
        Draw ECG paper-style grid (default ``False``).
    figsize : tuple
        Width is fixed; height is auto-calculated (2 in per row) when ``None``.
    share_x : bool
        Share the x-axis across all subplots (default ``True``).
    fs : int | None
        Sample rate in Hz.  Required when *leads* is a numpy array.
    show : bool
        Display the plot immediately (default ``True``).
    x_axis : str
        ``"time"`` for seconds (default) or ``"samples"`` for sample indices.
    rows : int | None
        Number of rows in the subplot grid. Derived from *cols* or defaults
        to one row per lead when neither is given.
    cols : int | None
        Number of columns in the subplot grid (default ``1``).
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    lead_list, rec = _resolve_leads(leads, fs=fs)
    if record is not None:
        rec = record

    n = len(lead_list)
    if n == 0:
        fig, _ = plt.subplots(figsize=(figsize[0], 3))
        return fig

    # Name leads built from numpy arrays; Lead objects are never modified
    if isinstance(leads, np.ndarray) or (
        isinstance(leads, list) and leads and isinstance(leads[0], np.ndarray)
    ):
        for i, ld in enumerate(lead_list[:len(STANDARD_12LEAD)]):
            ld.label = STANDARD_12LEAD[i]

    r, c = _grid_shape(n, rows, cols)
    h = figsize[1] if figsize[1] is not None else max(3, 2 * r)

    has_header = rec is not None
    if has_header:
        fig, all_axes = plt.subplots(
            r + 1, c, figsize=(figsize[0], h + 1.5),
            sharex=False, squeeze=False,
            gridspec_kw={"height_ratios": [0.4] + [1] * r},
        )
        # Merge top row into one header axes
        for ci in range(c):
            all_axes[0][ci].set_visible(False)
        ax_hdr = fig.add_subplot(r + 1, 1, 1)
        ax_hdr.axis("off")
        _draw_header(ax_hdr, rec)
        axes = all_axes[1:]
    else:
        fig, axes = plt.subplots(r, c, figsize=(figsize[0], h), sharex=share_x, squeeze=False)

    common = x_extent(lead_list, x_axis) if share_x else None
    decimated: list[str] = []
    for i, ld in enumerate(lead_list):
        ri, ci = divmod(i, c)
        ax = axes[ri][ci]
        x, _ = _x_data(ld, x_axis)
        _plot_trace(ax, ld, x, decimated, color=lead_color(ld.label), linewidth=0.8)
        unit = amplitude_unit(ld)
        ax.set_title(f"{ld.label} ({unit})" if unit else ld.label, fontsize=9, loc="left", pad=2)
        _finish_ax(ax, common or x_extent([ld], x_axis), show_grid, unit, x_axis)
    warn_decimated(decimated, _STATIC_STACKLEVEL)

    # Hide empty subplots
    for j in range(n, r * c):
        ri, ci = divmod(j, c)
        axes[ri][ci].set_visible(False)

    # X-axis label on bottom row
    for ci in range(c):
        axes[-1][ci].set_xlabel("Sample" if x_axis == "samples" else "Time (s)")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def _draw_header(ax, record: ECGRecord) -> None:
    """Draw patient/device/measurement info in the header axes."""
    lines = []
    p = record.patient
    name = f"{p.first_name} {p.last_name}".strip()
    if name:
        lines.append(f"Name: {name}")
    if p.patient_id:
        lines.append(f"ID: {p.patient_id}")
    parts = []
    if p.age is not None:
        parts.append(f"Age: {p.age}")
    if p.sex:
        parts.append(f"Sex: {p.sex}")
    if parts:
        lines.append("  ".join(parts))

    r = record.recording
    if r.date:
        lines.append(f"Date: {r.date.strftime('%Y-%m-%d %H:%M %z').strip()}")
    if r.acquisition.signal.sampling_rate:
        lines.append(f"Sample rate: {r.acquisition.signal.sampling_rate} Hz")

    d = r.device
    dev_parts = []
    if d.manufacturer:
        dev_parts.append(d.manufacturer)
    if d.model:
        dev_parts.append(d.model)
    if dev_parts:
        lines.append(f"Device: {' '.join(dev_parts)}")

    m = record.measurements
    meas = []
    if m.heart_rate is not None:
        meas.append(f"HR: {m.heart_rate} bpm")
    if m.pr_interval is not None:
        meas.append(f"PR: {m.pr_interval} ms")
    if m.qrs_duration is not None:
        meas.append(f"QRS: {m.qrs_duration} ms")
    if m.qt_interval is not None:
        meas.append(f"QT: {m.qt_interval} ms")
    if m.qtc_bazett is not None:
        meas.append(f"QTc: {m.qtc_bazett} ms")
    if m.qrs_axis is not None:
        meas.append(f"Axis: {m.qrs_axis}\u00b0")
    if meas:
        lines.append("  |  ".join(meas))

    interp = record.interpretation
    if interp.statements:
        stmts = [f"{left} {right}".strip() if right else left for left, right in interp.statements[:3]]
        lines.append("Interpretation: " + "; ".join(stmts))

    text = "\n".join(lines) if lines else "ECG Report"
    ax.text(
        0.02, 0.5, text, transform=ax.transAxes, fontsize=9,
        verticalalignment="center", fontfamily="monospace",
    )


@mpl_styled
def plot_peaks(
    lead: LeadLike,
    peaks: NDArray[np.intp] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 3),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
    show: bool = True,
    x_axis: str = "time",
) -> Figure:
    """Plot lead with R-peak markers and RR interval annotations.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        ECG lead or raw signal array to plot.
    peaks : NDArray | None
        R-peak indices. Auto-detected if ``None``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    show : bool
        Display the plot immediately (default ``True``).
    x_axis : str
        ``"time"`` for seconds (default) or ``"samples"`` for sample indices.
    """
    from ecgdatakit.processing.peaks import detect_r_peaks

    lead = ensure_lead(lead, fs=fs)
    check_rate(lead)
    if peaks is None:
        peaks = detect_r_peaks(lead)

    own_fig = ax is None
    fig, ax = _get_or_create_ax(figsize, ax)
    x, xlabel = _x_data(lead, x_axis)

    decimated: list[str] = []
    _plot_trace(ax, lead, x, decimated, color=lead_color(lead.label), linewidth=0.8)
    warn_decimated(decimated, _STATIC_STACKLEVEL)
    if len(peaks) > 0:
        ax.plot(x[peaks], lead.samples[peaks], "rv", markersize=7, label="R-peaks")

        for i in range(1, min(len(peaks), 20)):
            rr_ms = (peaks[i] - peaks[i - 1]) / lead.sampling_rate * 1000
            mid_x = (x[peaks[i]] + x[peaks[i - 1]]) / 2
            y_pos = max(lead.samples[peaks[i]], lead.samples[peaks[i - 1]])
            ax.annotate(
                f"{rr_ms:.0f}ms",
                xy=(mid_x, y_pos),
                fontsize=7,
                ha="center",
                va="bottom",
                color="#666666",
            )

        rr_all = np.diff(peaks).astype(np.float64) / lead.sampling_rate * 1000
        if len(rr_all) > 0:
            hr = 60_000.0 / rr_all.mean()
            ax.text(
                0.98, 0.95, f"HR: {hr:.0f} bpm",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(amplitude_label(lead))
    ax.set_title(title or f"{lead.label} \u2014 R-peaks")
    _finish_ax(ax, x_extent([lead], x_axis), True, amplitude_unit(lead), x_axis)
    fig.tight_layout()

    if show and own_fig:
        import matplotlib.pyplot as plt
        plt.show()

    return fig


@mpl_styled
def plot_beats(
    lead: LeadLike,
    beats: list[Lead] | None = None,
    peaks: NDArray[np.intp] | None = None,
    overlay: bool = True,
    figsize: tuple[float, float] = (8, 5),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
    show: bool = True,
    before: float = 0.2,
    after: float = 0.4,
) -> Figure:
    """Plot segmented heartbeats.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Source ECG lead or raw signal array.
    beats : list[Lead] | None
        Pre-segmented beats. Segmented automatically if ``None``. When given,
        they must have been cut with the same *before* window so the time
        axis is right.
    peaks : NDArray | None
        R-peak indices for segmentation.
    overlay : bool
        ``True``: overlay all beats with their mean; ``False``: waterfall display.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    show : bool
        Display the plot immediately (default ``True``).
    before, after : float
        Window in seconds before and after each R-peak (default 0.2, 0.4).
        Time 0 on the x-axis is the R-peak.
    """
    from ecgdatakit.processing.transforms import segment_beats

    lead = ensure_lead(lead, fs=fs)
    check_rate(lead)
    if beats is None:
        beats = segment_beats(lead, peaks, before, after)

    own_fig = ax is None
    fig, ax = _get_or_create_ax(figsize, ax)

    if not beats:
        ax.text(0.5, 0.5, "No beats detected", transform=ax.transAxes, ha="center")
        return fig

    n_samples = min(len(b.samples) for b in beats)
    pre = int(round(before * lead.sampling_rate))
    t_ms = (np.arange(n_samples, dtype=np.float64) - pre) / lead.sampling_rate * 1000

    if overlay:
        for beat in beats:
            ax.plot(t_ms, beat.samples[:n_samples], color=lead_color(lead.label),
                    alpha=0.25, linewidth=0.6)
        avg = np.mean([b.samples[:n_samples] for b in beats], axis=0)
        ax.plot(t_ms, avg, color="black", linewidth=2.0, label="Average")
        ax.legend(fontsize=8)
    else:
        offset = 0.0
        spacing = np.ptp(beats[0].samples) * 1.3 if n_samples > 0 else 1.0
        for beat in beats:
            ax.plot(t_ms, beat.samples[:n_samples] + offset, color=lead_color(lead.label),
                    linewidth=0.7)
            offset -= spacing

    ax.set_xlabel("Time relative to R-peak (ms)")
    ax.set_ylabel(amplitude_label(lead))
    ax.set_title(f"{lead.label} \u2014 Segmented beats ({len(beats)})")
    _style_ax(ax)
    fig.tight_layout()

    if show and own_fig:
        import matplotlib.pyplot as plt
        plt.show()

    return fig


@mpl_styled
def plot_average_beat(
    lead: LeadLike,
    peaks: NDArray[np.intp] | None = None,
    before: float = 0.2,
    after: float = 0.4,
    figsize: tuple[float, float] = (6, 4),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
    show: bool = True,
) -> Figure:
    """Plot ensemble-averaged beat with \u00b11 SD shading.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Source ECG lead or raw signal array.
    peaks : NDArray | None
        R-peak indices.
    before : float
        Seconds before R-peak.
    after : float
        Seconds after R-peak.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    show : bool
        Display the plot immediately (default ``True``).
    """
    from ecgdatakit.processing.transforms import segment_beats

    lead = ensure_lead(lead, fs=fs)
    check_rate(lead)
    beats = segment_beats(lead, peaks, before, after)

    own_fig = ax is None
    fig, ax = _get_or_create_ax(figsize, ax)

    if not beats:
        ax.text(0.5, 0.5, "No beats detected", transform=ax.transAxes, ha="center")
        return fig

    stacked = np.stack([b.samples for b in beats], axis=0)
    avg = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    pre = int(round(before * lead.sampling_rate))
    t_ms = (np.arange(len(avg), dtype=np.float64) - pre) / lead.sampling_rate * 1000

    ax.fill_between(t_ms, avg - std, avg + std, alpha=0.25, color=lead_color(lead.label))
    ax.plot(t_ms, avg, color=lead_color(lead.label), linewidth=2.0)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="R-peak")

    ax.set_xlabel("Time relative to R-peak (ms)")
    ax.set_ylabel(amplitude_label(lead))
    ax.set_title(f"{lead.label} \u2014 Average beat (n={len(beats)})")
    ax.legend(fontsize=8)
    _style_ax(ax)
    fig.tight_layout()

    if show and own_fig:
        import matplotlib.pyplot as plt
        plt.show()

    return fig


@mpl_styled
def plot_spectrum(
    lead: LeadLike,
    method: str = "welch",
    figsize: tuple[float, float] = (10, 4),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
    show: bool = True,
) -> Figure:
    """Plot power spectral density or FFT magnitude spectrum.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        ECG lead or raw signal array to analyse.
    method : str
        ``"welch"`` for PSD or ``"fft"`` for magnitude spectrum.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    show : bool
        Display the plot immediately (default ``True``).
    """
    lead = ensure_lead(lead, fs=fs)
    check_rate(lead)
    from ecgdatakit.processing.transforms import fft as ecg_fft
    from ecgdatakit.processing.transforms import power_spectrum

    own_fig = ax is None
    fig, ax = _get_or_create_ax(figsize, ax)

    if method == "welch":
        freqs, power = power_spectrum(lead)
        power_db = 10 * np.log10(np.maximum(power, 1e-20))
        ax.plot(freqs, power_db, color=lead_color(lead.label), linewidth=0.8)
        ax.set_ylabel("Power (dB/Hz)")
    else:
        freqs, mags = ecg_fft(lead)
        ax.plot(freqs, mags, color=lead_color(lead.label), linewidth=0.8)
        ax.set_ylabel("Magnitude")

    ax.axvspan(0.05, 150, alpha=0.05, color="green", label="ECG band (0.05\u2013150 Hz)")
    ax.axvspan(0, 0.05, alpha=0.05, color="red", label="Baseline drift")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"{lead.label} \u2014 {'PSD (Welch)' if method == 'welch' else 'FFT Magnitude'}")
    ax.set_xlim(0, min(lead.sampling_rate / 2, 250))
    ax.legend(fontsize=7, loc="upper right")
    _style_ax(ax)
    fig.tight_layout()

    if show and own_fig:
        import matplotlib.pyplot as plt
        plt.show()

    return fig


@mpl_styled
def plot_spectrogram(
    lead: LeadLike,
    nperseg: int = 256,
    figsize: tuple[float, float] = (12, 4),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
    show: bool = True,
) -> Figure:
    """Plot time-frequency spectrogram (STFT).

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        ECG lead or raw signal array.
    nperseg : int
        Segment length for STFT.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    show : bool
        Display the plot immediately (default ``True``).
    """
    from ecgdatakit.processing._core import require_scipy

    lead = ensure_lead(lead, fs=fs)
    check_rate(lead)
    sig = require_scipy("signal")
    own_fig = ax is None
    fig, ax = _get_or_create_ax(figsize, ax)

    nperseg = min(nperseg, len(lead.samples))
    f, t_spec, Sxx = sig.spectrogram(
        lead.samples, fs=lead.sampling_rate, nperseg=nperseg
    )
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-20))

    ax.pcolormesh(t_spec, f, Sxx_db, shading="gouraud", cmap="viridis")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{lead.label} \u2014 Spectrogram")
    ax.set_ylim(0, min(lead.sampling_rate / 2, 150))
    _style_ax(ax)
    fig.tight_layout()

    if show and own_fig:
        import matplotlib.pyplot as plt
        plt.show()

    return fig


@mpl_styled
def plot_rr_tachogram(
    rr_ms: NDArray[np.float64],
    figsize: tuple[float, float] = (10, 3),
    ax: Axes | None = None,
    *,
    show: bool = True,
) -> Figure:
    """Plot RR interval tachogram.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.
    show : bool
        Display the plot immediately (default ``True``).
    """
    rr_ms = check_rr(rr_ms)
    own_fig = ax is None
    fig, ax = _get_or_create_ax(figsize, ax)

    beats = np.arange(len(rr_ms))
    ax.plot(beats, rr_ms, color="#1f77b4", linewidth=0.8, marker=".", markersize=3)

    mean_rr = rr_ms.mean()
    std_rr = rr_ms.std()
    ax.axhline(mean_rr, color="red", linestyle="--", linewidth=0.8, label=f"Mean: {mean_rr:.0f} ms")
    ax.axhline(mean_rr + std_rr, color="orange", linestyle=":", linewidth=0.6, label=f"\u00b1SD: {std_rr:.0f} ms")
    ax.axhline(mean_rr - std_rr, color="orange", linestyle=":", linewidth=0.6)

    ax.set_xlabel("Beat number")
    ax.set_ylabel("RR interval (ms)")
    ax.set_title("RR Tachogram")
    ax.legend(fontsize=8)
    _style_ax(ax)
    fig.tight_layout()

    if show and own_fig:
        import matplotlib.pyplot as plt
        plt.show()

    return fig


@mpl_styled
def plot_poincare(
    rr_ms: NDArray[np.float64],
    figsize: tuple[float, float] = (6, 6),
    ax: Axes | None = None,
    *,
    show: bool = True,
) -> Figure:
    """Poincar\u00e9 plot: RR(n) vs RR(n+1) with SD1/SD2 ellipse.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.
    show : bool
        Display the plot immediately (default ``True``).
    """
    from matplotlib.patches import Ellipse

    from ecgdatakit.processing.hrv import poincare

    rr_ms = check_rr(rr_ms)
    own_fig = ax is None
    fig, ax = _get_or_create_ax(figsize, ax)

    if len(rr_ms) < 2:
        ax.text(0.5, 0.5, "Need \u22652 RR intervals", transform=ax.transAxes, ha="center")
        return fig

    x = rr_ms[:-1]
    y = rr_ms[1:]

    ax.scatter(x, y, s=10, alpha=0.5, color="#1f77b4", edgecolors="none")

    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    margin = (hi - lo) * 0.1
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=0.5, alpha=0.4)

    # SD1/SD2 from the HRV module so the plot and the metrics agree
    desc = poincare(rr_ms)
    sd1, sd2 = desc["sd1"], desc["sd2"]
    if np.isfinite(sd1) and np.isfinite(sd2):
        ellipse = Ellipse(
            (float(x.mean()), float(y.mean())), width=2 * sd2, height=2 * sd1, angle=45,
            edgecolor="red", facecolor="none", linewidth=1.5, linestyle="--",
            label=f"SD1={sd1:.1f}, SD2={sd2:.1f}",
        )
        ax.add_patch(ellipse)

    ax.set_xlabel("RR(n) (ms)")
    ax.set_ylabel("RR(n+1) (ms)")
    ax.set_title("Poincar\u00e9 Plot")
    ax.set_aspect("equal", adjustable="datalim")
    if ax.patches:
        ax.legend(fontsize=8)
    _style_ax(ax)
    fig.tight_layout()

    if show and own_fig:
        import matplotlib.pyplot as plt
        plt.show()

    return fig


@mpl_styled
def plot_hrv_summary(
    rr_ms: NDArray[np.float64],
    figsize: tuple[float, float] = (14, 8),
    *,
    show: bool = True,
) -> Figure:
    """Combined HRV dashboard: tachogram, Poincar\u00e9, frequency bands, metrics.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.
    show : bool
        Display the plot immediately (default ``True``).
    """
    import matplotlib.pyplot as plt

    rr_ms = check_rr(rr_ms)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_rr_tachogram(rr_ms, ax=axes[0, 0], show=False)

    plot_poincare(rr_ms, ax=axes[0, 1], show=False)

    _plot_hrv_frequency(rr_ms, ax=axes[1, 0])

    _plot_hrv_table(rr_ms, ax=axes[1, 1])

    # Style the HRV frequency axis (sub-functions styled their own axes)
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    fig.suptitle("HRV Summary", fontsize=14, y=1.01)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def _plot_hrv_frequency(rr_ms: NDArray[np.float64], ax) -> None:
    """Bar chart of the VLF/LF/HF band powers from :func:`frequency_domain`.

    The values come from the HRV module, so the plot always matches the
    metrics it reports.
    """
    from ecgdatakit.processing.hrv import frequency_domain

    # Short-recording warnings are re-emitted at the caller of plot_hrv_summary
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fd = frequency_domain(rr_ms)
    for w in caught:
        warnings.warn(w.message, w.category, stacklevel=_STATIC_STACKLEVEL)
    if not np.isfinite(fd["total_power"]):
        ax.text(0.5, 0.5, "Not enough RR intervals for the spectrum",
                transform=ax.transAxes, ha="center")
        ax.axis("off")
        return

    names = ["VLF\n0-0.04 Hz", "LF\n0.04-0.15 Hz", "HF\n0.15-0.40 Hz"]
    values = [fd["vlf_power"], fd["lf_power"], fd["hf_power"]]
    bars = ax.bar(names, values, color=["#9467bd", "#2ca02c", "#1f77b4"], alpha=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}",
                ha="center", va="bottom", fontsize=8)
    ratio = fd["lf_hf_ratio"]
    ax.set_ylabel("Power (ms\u00b2)")
    ax.set_title("HRV Frequency Bands" + (f" (LF/HF {ratio:.2f})" if np.isfinite(ratio) else ""))


def _plot_hrv_table(rr_ms: NDArray[np.float64], ax) -> None:
    """Draw a table of time-domain HRV metrics."""
    from ecgdatakit.processing.hrv import time_domain

    metrics = time_domain(rr_ms)

    rows = [
        ("Mean RR", f"{metrics['mean_rr']:.1f} ms"),
        ("SDNN", f"{metrics['sdnn']:.1f} ms"),
        ("RMSSD", f"{metrics['rmssd']:.1f} ms"),
        ("pNN50", f"{metrics['pnn50']:.1f} %"),
        ("pNN20", f"{metrics['pnn20']:.1f} %"),
        ("Mean HR", f"{metrics['hr_mean']:.1f} bpm"),
        ("HR Std", f"{metrics['hr_std']:.1f} bpm"),
    ]

    ax.axis("off")
    table = ax.table(
        cellText=[[r[0], r[1]] for r in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Time-Domain Metrics", fontsize=11, pad=10)


@mpl_styled
def plot_quality(
    leads: list[Lead] | ECGRecord | NDArray[np.float64] | list[NDArray[np.float64]],
    figsize: tuple[float, float] = (10, 5),
    *,
    fs: int | None = None,
    show: bool = True,
    seconds: float | None = None,
) -> Figure:
    """Signal quality dashboard: SQI bar chart per lead.

    The quality metrics run over the whole of each lead by default, which
    takes a few seconds per lead on a 24 h Holter. Pass *seconds* to assess
    only the start of each lead.

    Parameters
    ----------
    leads : list[Lead] | ECGRecord | NDArray | list[NDArray]
        Leads to assess.  Also accepts a 2-D numpy array
        (n_leads × n_samples) or a list of 1-D numpy arrays.
    fs : int | None
        Sample rate in Hz.  Required when *leads* is a numpy array.
    show : bool
        Display the plot immediately (default ``True``).
    seconds : float | None
        Assess only the first *seconds* of each lead (default: whole lead).
    """
    require_matplotlib()
    import matplotlib.pyplot as plt
    from ecgdatakit.processing.quality import signal_quality_index, snr_estimate

    lead_list, _ = _resolve_leads(leads, fs=fs)
    if not lead_list:
        fig, _ = plt.subplots(figsize=figsize)
        return fig

    if seconds is not None:
        lead_list = [_first_seconds(ld, seconds) for ld in lead_list]
    labels = [ld.label for ld in lead_list]
    sqis = [signal_quality_index(ld) for ld in lead_list]
    snrs = [snr_estimate(ld) for ld in lead_list]

    colors = []
    for s in sqis:
        if s > 0.8:
            colors.append("#2ca02c")
        elif s >= 0.5:
            colors.append("#ff7f0e")
        else:
            colors.append("#d62728")

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, sqis, color=colors, edgecolor="white", linewidth=0.5)

    for i, (bar, snr) in enumerate(zip(bars, snrs)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"SNR: {snr:.0f} dB",
            ha="center", va="bottom", fontsize=8, color="#555555",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Signal Quality Index")
    ax.set_ylim(0, 1.15)
    ax.set_title("Signal Quality per Lead"
                 + (f" (first {seconds:g} s)" if seconds is not None else ""))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Excellent (>0.8)"),
        Patch(facecolor="#ff7f0e", label="Acceptable (0.5\u20130.8)"),
        Patch(facecolor="#d62728", label="Unacceptable (<0.5)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if show:
        plt.show()

    return fig



_REPORT_SECONDS = 10.0
_COLUMN_SECONDS = 2.5


def _first_seconds(lead: Lead, seconds: float) -> Lead:
    """The first *seconds* of a lead, as a new Lead."""
    import dataclasses

    check_rate(lead)
    n = int(round(seconds * lead.sampling_rate))
    return dataclasses.replace(lead, samples=lead.samples[:n])


@mpl_styled
def plot_report(
    record: ECGRecord,
    figsize: tuple[float, float] = (16, 20),
    *,
    show: bool = True,
) -> Figure:
    """Comprehensive ECG report page.

    Includes a header with patient/device info, the standard 3 x 4 layout
    (each column shows its own 2.5 s of the first 10 s: I/II/III from 0 to
    2.5 s, aVR/aVL/aVF from 2.5 to 5 s, and so on), a 10 s lead II rhythm
    strip, and signal quality for the same first 10 s. Longer recordings
    (e.g. Holter) are not drawn beyond the first 10 s; use :func:`plot_lead`
    on a slice to look elsewhere.

    The paper grid uses 0.2 s / 0.5 mV squares when the leads are in a
    voltage unit. Leads in raw counts are drawn with automatic amplitude
    ticks and labelled "raw counts".

    Parameters
    ----------
    record : ECGRecord
        Full ECG record.
    show : bool
        Display the plot immediately (default ``True``).
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(6, 4, figure=fig, height_ratios=[0.6, 1, 1, 1, 0.7, 0.8], hspace=0.35, wspace=0.15)

    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.axis("off")
    _draw_header(ax_hdr, record)

    shown = [_first_seconds(ld, _REPORT_SECONDS) for ld in record.leads]
    for row_idx, row_labels in enumerate(GRID_12LEAD):
        for col_idx, lbl in enumerate(row_labels):
            ax = fig.add_subplot(gs[1 + row_idx, col_idx])
            t0 = col_idx * _COLUMN_SECONDS
            t1 = t0 + _COLUMN_SECONDS
            ld = _find_lead(shown, lbl)
            unit = ""
            if ld is not None:
                t = time_axis(ld)
                sl = (t >= t0) & (t <= t1)
                ax.plot(t[sl], ld.samples[sl], color=lead_color(lbl), linewidth=0.7)
                unit = amplitude_unit(ld)
            ax.set_title(f"{lbl} ({unit})" if unit else lbl, fontsize=9, loc="left", pad=2)
            ax.set_xlim(t0, t1)
            _style_ax(ax)
            _ecg_grid(ax, unit)
            ax.tick_params(labelsize=6)
            if row_idx < 2:
                ax.tick_params(labelbottom=False)

    ax_rhythm = fig.add_subplot(gs[4, :])
    rl = _find_lead(shown, "II")
    unit = ""
    if rl is not None and len(rl.samples):
        t = time_axis(rl)
        ax_rhythm.plot(t, rl.samples, color=lead_color("II"), linewidth=0.7)
        unit = amplitude_unit(rl)
    ax_rhythm.set_xlim(0, _REPORT_SECONDS)
    ax_rhythm.set_title(f"II rhythm strip ({unit})" if unit else "II rhythm strip",
                        fontsize=9, loc="left", pad=2)
    ax_rhythm.set_xlabel("Time (s)", fontsize=8)
    _style_ax(ax_rhythm)
    _ecg_grid(ax_rhythm, unit)
    ax_rhythm.tick_params(labelsize=6)

    ax_qi = fig.add_subplot(gs[5, :2])
    ax_qi.axis("off")
    _draw_quality_summary(ax_qi, shown)

    ax_interp = fig.add_subplot(gs[5, 2:])
    ax_interp.axis("off")
    _draw_interpretation(ax_interp, record)

    # The header and text panels are not tight_layout compatible
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()

    if show:
        plt.show()

    return fig


def _draw_quality_summary(ax, leads: list[Lead]) -> None:
    """Draw compact quality summary text for the plotted window."""
    from ecgdatakit.processing.quality import classify_quality, signal_quality_index

    lines = [f"Signal Quality (first {_REPORT_SECONDS:.0f} s):"]
    for ld in leads[:12]:
        if len(ld.samples) == 0:
            continue
        sqi = signal_quality_index(ld)
        cat = classify_quality(ld)
        lines.append(f"  {ld.label:>5}: {sqi:.2f} ({cat})")

    ax.text(
        0.02, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
    )


def _draw_interpretation(ax, record: ECGRecord) -> None:
    """Draw interpretation statements."""
    interp = record.interpretation
    lines = ["Interpretation:"]
    if interp.severity:
        lines.append(f"  Severity: {interp.severity}")
    if interp.source:
        lines.append(f"  Source: {interp.source}")
    for left, right in interp.statements:
        text = f"{left} {right}".strip() if right else left
        lines.append(f"  - {text}")
    if not interp.statements and not interp.severity:
        lines.append("  No interpretation available")

    ax.text(
        0.02, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
    )