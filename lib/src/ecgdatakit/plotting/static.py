"""Static ECG plots using matplotlib.

All public functions return ``matplotlib.figure.Figure``.
Functions that accept an *ax* parameter can render into an existing axes
for composability; when *ax* is ``None`` a new figure is created.

Requires: ``pip install ecgdatakit[plotting]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import ECGRecord, Lead, LeadLike
from ecgdatakit.plotting._core import (
    GRID_12LEAD,
    _find_lead,
    _resolve_leads,
    ensure_lead,
    lead_color,
    require_matplotlib,
    time_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure



def _get_or_create_ax(figsize, ax):
    """Return (fig, ax).  Creates new ones when *ax* is ``None``."""
    mpl = require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _ecg_grid(ax, major_x=0.2, major_y=0.5, minor_x=0.04, minor_y=0.1):
    """Draw ECG paper-style grid on *ax*."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#ffcccc", linewidth=0.8)
    ax.grid(True, which="minor", color="#ffe6e6", linewidth=0.4)

    from matplotlib.ticker import MultipleLocator

    ax.xaxis.set_major_locator(MultipleLocator(major_x))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_x))
    ax.yaxis.set_major_locator(MultipleLocator(major_y))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_y))



def plot_lead(
    lead: LeadLike,
    peaks: NDArray[np.intp] | None = None,
    title: str | None = None,
    show_grid: bool = True,
    figsize: tuple[float, float] = (12, 3),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
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
        Draw ECG paper-style grid (default ``True``).
    figsize : tuple
        Figure size in inches (default ``(12, 3)``).
    ax : Axes | None
        Existing axes to draw on. A new figure is created if ``None``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    fig, ax = _get_or_create_ax(figsize, ax)
    t = time_axis(lead)

    ax.plot(t, lead.samples, color=lead_color(lead.label), linewidth=0.8)

    if peaks is not None and len(peaks) > 0:
        ax.plot(
            t[peaks],
            lead.samples[peaks],
            "rv",
            markersize=6,
            label="R-peaks",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Amplitude ({lead.units})" if lead.units else "Amplitude")
    ax.set_title(title or lead.label)

    if show_grid:
        _ecg_grid(ax)

    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    return fig


def plot_leads(
    leads: list[Lead] | ECGRecord,
    peaks_dict: dict[str, NDArray[np.intp]] | None = None,
    title: str | None = None,
    show_grid: bool = True,
    figsize: tuple[float, float | None] = (12, None),
    share_x: bool = True,
) -> Figure:
    """Plot multiple leads stacked vertically.

    Parameters
    ----------
    leads : list[Lead] | ECGRecord
        Leads to plot.
    peaks_dict : dict | None
        ``{label: peaks_array}`` for per-lead R-peak markers.
    title : str | None
        Overall figure title.
    show_grid : bool
        Draw ECG paper-style grid.
    figsize : tuple
        Width is fixed; height is auto-calculated (2 in per lead) when ``None``.
    share_x : bool
        Share the x-axis across all subplots (default ``True``).
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    lead_list, _ = _resolve_leads(leads)
    n = len(lead_list)
    if n == 0:
        fig, _ = plt.subplots(figsize=(figsize[0], 3))
        return fig

    h = figsize[1] if figsize[1] is not None else max(3, 2 * n)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], h), sharex=share_x)
    if n == 1:
        axes = [axes]

    for i, ld in enumerate(lead_list):
        ax = axes[i]
        t = time_axis(ld)
        ax.plot(t, ld.samples, color=lead_color(ld.label), linewidth=0.8)

        if peaks_dict and ld.label in peaks_dict:
            pk = peaks_dict[ld.label]
            ax.plot(t[pk], ld.samples[pk], "rv", markersize=5)

        ax.set_ylabel(ld.label, rotation=0, labelpad=30, fontsize=10)
        ax.yaxis.set_label_position("left")
        if show_grid:
            _ecg_grid(ax)
        ax.set_xlim(t[0], t[-1])

    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_12lead(
    leads: list[Lead] | ECGRecord,
    record: ECGRecord | None = None,
    paper_speed: float = 25,
    amplitude: float = 10,
    rhythm_lead: str = "II",
    duration: float = 10.0,
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """Standard 12-lead ECG grid layout.

    Parameters
    ----------
    leads : list[Lead] | ECGRecord
        Leads (or full record) to plot.
    record : ECGRecord | None
        If provided, header with patient/device/measurement info is shown.
    paper_speed : float
        Paper speed in mm/s (default 25).
    amplitude : float
        Amplitude scale in mm/mV (default 10).
    rhythm_lead : str
        Lead used for the full-length rhythm strip (default ``"II"``).
    duration : float
        Seconds of signal to show per cell (default 10.0).
    figsize : tuple
        Figure size.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    lead_list, rec = _resolve_leads(leads)
    if record is not None:
        rec = record

    has_header = rec is not None
    nrows = 5 if has_header else 4
    height_ratios = [0.8, 1, 1, 1, 0.8] if has_header else [1, 1, 1, 0.8]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        nrows, 4, figure=fig, height_ratios=height_ratios, hspace=0.3, wspace=0.15
    )

    row_offset = 1 if has_header else 0

    if has_header and rec is not None:
        ax_hdr = fig.add_subplot(gs[0, :])
        ax_hdr.axis("off")
        _draw_header(ax_hdr, rec)

    for row_idx, row_labels in enumerate(GRID_12LEAD):
        for col_idx, lbl in enumerate(row_labels):
            ax = fig.add_subplot(gs[row_offset + row_idx, col_idx])
            ld = _find_lead(lead_list, lbl)
            if ld is not None:
                t = time_axis(ld)
                max_samples = int(duration * ld.sample_rate)
                sl = slice(0, min(max_samples, len(ld.samples)))
                ax.plot(
                    t[sl], ld.samples[sl], color=lead_color(lbl), linewidth=0.7
                )
                ax.set_xlim(0, duration)
            ax.set_title(lbl, fontsize=9, loc="left", pad=2)
            _ecg_grid(ax)
            ax.tick_params(labelsize=7)
            if row_idx < 2:
                ax.set_xticklabels([])

    ax_rhythm = fig.add_subplot(gs[row_offset + 3, :])
    rl = _find_lead(lead_list, rhythm_lead)
    if rl is not None:
        t = time_axis(rl)
        ax_rhythm.plot(t, rl.samples, color=lead_color(rhythm_lead), linewidth=0.7)
        ax_rhythm.set_xlim(t[0], t[-1])
    ax_rhythm.set_title(f"{rhythm_lead} rhythm strip", fontsize=9, loc="left", pad=2)
    _ecg_grid(ax_rhythm)
    ax_rhythm.set_xlabel("Time (s)", fontsize=8)
    ax_rhythm.tick_params(labelsize=7)

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
        lines.append(f"Date: {r.date.strftime('%Y-%m-%d %H:%M')}")
    if r.sample_rate:
        lines.append(f"Sample rate: {r.sample_rate} Hz")

    d = record.device
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
        meas.append(f"Axis: {m.qrs_axis}°")
    if meas:
        lines.append("  |  ".join(meas))

    interp = record.interpretation
    if interp.statements:
        lines.append("Interpretation: " + "; ".join(interp.statements[:3]))

    text = "\n".join(lines) if lines else "ECG Report"
    ax.text(
        0.02, 0.5, text, transform=ax.transAxes, fontsize=9,
        verticalalignment="center", fontfamily="monospace",
    )



def plot_peaks(
    lead: LeadLike,
    peaks: NDArray[np.intp] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 3),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
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
    """
    from ecgdatakit.processing.peaks import detect_r_peaks

    lead = ensure_lead(lead, fs=fs)
    if peaks is None:
        peaks = detect_r_peaks(lead)

    fig, ax = _get_or_create_ax(figsize, ax)
    t = time_axis(lead)

    ax.plot(t, lead.samples, color=lead_color(lead.label), linewidth=0.8)
    if len(peaks) > 0:
        ax.plot(t[peaks], lead.samples[peaks], "rv", markersize=7, label="R-peaks")

        for i in range(1, min(len(peaks), 20)):
            rr_ms = (peaks[i] - peaks[i - 1]) / lead.sample_rate * 1000
            mid_t = (t[peaks[i]] + t[peaks[i - 1]]) / 2
            y_pos = max(lead.samples[peaks[i]], lead.samples[peaks[i - 1]])
            ax.annotate(
                f"{rr_ms:.0f}ms",
                xy=(mid_t, y_pos),
                fontsize=7,
                ha="center",
                va="bottom",
                color="#666666",
            )

        rr_all = np.diff(peaks).astype(np.float64) / lead.sample_rate * 1000
        if len(rr_all) > 0:
            hr = 60_000.0 / rr_all.mean()
            ax.text(
                0.98, 0.95, f"HR: {hr:.0f} bpm",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Amplitude ({lead.units})" if lead.units else "Amplitude")
    ax.set_title(title or f"{lead.label} — R-peaks")
    _ecg_grid(ax)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    return fig


def plot_beats(
    lead: LeadLike,
    beats: list[Lead] | None = None,
    peaks: NDArray[np.intp] | None = None,
    overlay: bool = True,
    figsize: tuple[float, float] = (8, 5),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
) -> Figure:
    """Plot segmented heartbeats.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Source ECG lead or raw signal array.
    beats : list[Lead] | None
        Pre-segmented beats. Segmented automatically if ``None``.
    peaks : NDArray | None
        R-peak indices for segmentation.
    overlay : bool
        ``True``: overlay all beats; ``False``: waterfall display.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    from ecgdatakit.processing.transforms import average_beat, segment_beats

    lead = ensure_lead(lead, fs=fs)
    if beats is None:
        beats = segment_beats(lead, peaks)

    fig, ax = _get_or_create_ax(figsize, ax)

    if not beats:
        ax.text(0.5, 0.5, "No beats detected", transform=ax.transAxes, ha="center")
        return fig

    n_samples = len(beats[0].samples)
    t_ms = np.arange(n_samples, dtype=np.float64) / lead.sample_rate * 1000

    if overlay:
        for i, beat in enumerate(beats):
            ax.plot(t_ms, beat.samples, color=lead_color(lead.label), alpha=0.25, linewidth=0.6)
        avg = average_beat(lead, peaks)
        ax.plot(t_ms[:len(avg.samples)], avg.samples, color="black", linewidth=2.0, label="Average")
        ax.legend(fontsize=8)
    else:
        offset = 0.0
        spacing = np.ptp(beats[0].samples) * 1.3 if len(beats[0].samples) > 0 else 1.0
        for i, beat in enumerate(beats):
            ax.plot(t_ms, beat.samples + offset, color=lead_color(lead.label), linewidth=0.7)
            offset -= spacing

    ax.set_xlabel("Time relative to R-peak (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{lead.label} — Segmented beats ({len(beats)})")
    fig.tight_layout()
    return fig


def plot_average_beat(
    lead: LeadLike,
    peaks: NDArray[np.intp] | None = None,
    before: float = 0.2,
    after: float = 0.4,
    figsize: tuple[float, float] = (6, 4),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
) -> Figure:
    """Plot ensemble-averaged beat with ±1 SD shading.

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
    """
    from ecgdatakit.processing.transforms import segment_beats

    lead = ensure_lead(lead, fs=fs)
    beats = segment_beats(lead, peaks, before, after)

    fig, ax = _get_or_create_ax(figsize, ax)

    if not beats:
        ax.text(0.5, 0.5, "No beats detected", transform=ax.transAxes, ha="center")
        return fig

    stacked = np.stack([b.samples for b in beats], axis=0)
    avg = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    n_samples = len(avg)
    t_ms = np.linspace(-before * 1000, after * 1000, n_samples)

    ax.fill_between(t_ms, avg - std, avg + std, alpha=0.25, color=lead_color(lead.label))
    ax.plot(t_ms, avg, color=lead_color(lead.label), linewidth=2.0)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="R-peak")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{lead.label} — Average beat (n={len(beats)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig



def plot_spectrum(
    lead: LeadLike,
    method: str = "welch",
    figsize: tuple[float, float] = (10, 4),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
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
    """
    lead = ensure_lead(lead, fs=fs)
    from ecgdatakit.processing.transforms import fft as ecg_fft
    from ecgdatakit.processing.transforms import power_spectrum

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

    ax.axvspan(0.05, 150, alpha=0.05, color="green", label="ECG band (0.05–150 Hz)")
    ax.axvspan(0, 0.05, alpha=0.05, color="red", label="Baseline drift")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"{lead.label} — {'PSD (Welch)' if method == 'welch' else 'FFT Magnitude'}")
    ax.set_xlim(0, min(lead.sample_rate / 2, 250))
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def plot_spectrogram(
    lead: LeadLike,
    nperseg: int = 256,
    figsize: tuple[float, float] = (12, 4),
    ax: Axes | None = None,
    *,
    fs: int | None = None,
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
    """
    from ecgdatakit.processing._core import require_scipy

    lead = ensure_lead(lead, fs=fs)
    sig = require_scipy("signal")
    fig, ax = _get_or_create_ax(figsize, ax)

    nperseg = min(nperseg, len(lead.samples))
    f, t_spec, Sxx = sig.spectrogram(
        lead.samples, fs=lead.sample_rate, nperseg=nperseg
    )
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-20))

    ax.pcolormesh(t_spec, f, Sxx_db, shading="gouraud", cmap="viridis")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{lead.label} — Spectrogram")
    ax.set_ylim(0, min(lead.sample_rate / 2, 150))
    fig.tight_layout()
    return fig



def plot_rr_tachogram(
    rr_ms: NDArray[np.float64],
    figsize: tuple[float, float] = (10, 3),
    ax: Axes | None = None,
) -> Figure:
    """Plot RR interval tachogram.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.
    """
    fig, ax = _get_or_create_ax(figsize, ax)

    beats = np.arange(len(rr_ms))
    ax.plot(beats, rr_ms, color="#1f77b4", linewidth=0.8, marker=".", markersize=3)

    mean_rr = rr_ms.mean()
    std_rr = rr_ms.std()
    ax.axhline(mean_rr, color="red", linestyle="--", linewidth=0.8, label=f"Mean: {mean_rr:.0f} ms")
    ax.axhline(mean_rr + std_rr, color="orange", linestyle=":", linewidth=0.6, label=f"±SD: {std_rr:.0f} ms")
    ax.axhline(mean_rr - std_rr, color="orange", linestyle=":", linewidth=0.6)

    ax.set_xlabel("Beat number")
    ax.set_ylabel("RR interval (ms)")
    ax.set_title("RR Tachogram")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_poincare(
    rr_ms: NDArray[np.float64],
    figsize: tuple[float, float] = (6, 6),
    ax: Axes | None = None,
) -> Figure:
    """Poincaré plot: RR(n) vs RR(n+1) with SD1/SD2 ellipse.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.
    """
    from matplotlib.patches import Ellipse

    fig, ax = _get_or_create_ax(figsize, ax)

    if len(rr_ms) < 2:
        ax.text(0.5, 0.5, "Need ≥2 RR intervals", transform=ax.transAxes, ha="center")
        return fig

    x = rr_ms[:-1]
    y = rr_ms[1:]

    ax.scatter(x, y, s=10, alpha=0.5, color="#1f77b4", edgecolors="none")

    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    margin = (hi - lo) * 0.1
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=0.5, alpha=0.4)

    sd1 = float(np.std(y - x, ddof=1) / np.sqrt(2))
    sd2 = float(np.std(y + x, ddof=1) / np.sqrt(2))
    cx, cy = float(x.mean()), float(y.mean())

    ellipse = Ellipse(
        (cx, cy), width=2 * sd2, height=2 * sd1, angle=45,
        edgecolor="red", facecolor="none", linewidth=1.5, linestyle="--",
        label=f"SD1={sd1:.1f}, SD2={sd2:.1f}",
    )
    ax.add_patch(ellipse)

    ax.set_xlabel("RR(n) (ms)")
    ax.set_ylabel("RR(n+1) (ms)")
    ax.set_title("Poincaré Plot")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_hrv_summary(
    rr_ms: NDArray[np.float64],
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """Combined HRV dashboard: tachogram, Poincaré, frequency bands, metrics.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_rr_tachogram(rr_ms, ax=axes[0, 0])

    plot_poincare(rr_ms, ax=axes[0, 1])

    _plot_hrv_frequency(rr_ms, ax=axes[1, 0])

    _plot_hrv_table(rr_ms, ax=axes[1, 1])

    fig.suptitle("HRV Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def _plot_hrv_frequency(rr_ms: NDArray[np.float64], ax) -> None:
    """Plot HRV frequency-domain PSD with shaded VLF/LF/HF bands."""
    from ecgdatakit.processing._core import require_scipy

    if len(rr_ms) < 4:
        ax.text(0.5, 0.5, "Need ≥4 RR intervals", transform=ax.transAxes, ha="center")
        return

    sig = require_scipy("signal")
    interpolate = require_scipy("interpolate")

    rr_s = rr_ms / 1000.0
    t_rr = np.cumsum(rr_s) - rr_s[0]
    interp_fs = 4.0
    t_uniform = np.arange(0, t_rr[-1], 1.0 / interp_fs)
    f_interp = interpolate.interp1d(t_rr, rr_ms, kind="cubic", fill_value="extrapolate")
    rr_uniform = f_interp(t_uniform)
    rr_uniform = rr_uniform - rr_uniform.mean()

    nperseg = min(256, len(rr_uniform))
    freqs, psd = sig.welch(rr_uniform, fs=interp_fs, nperseg=nperseg)

    ax.plot(freqs, psd, color="black", linewidth=0.8)
    ax.fill_between(freqs, psd, where=(freqs < 0.04), alpha=0.3, color="#9467bd", label="VLF")
    ax.fill_between(freqs, psd, where=(freqs >= 0.04) & (freqs < 0.15), alpha=0.3, color="#2ca02c", label="LF")
    ax.fill_between(freqs, psd, where=(freqs >= 0.15) & (freqs < 0.40), alpha=0.3, color="#1f77b4", label="HF")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (ms²/Hz)")
    ax.set_title("HRV Frequency Domain")
    ax.set_xlim(0, 0.5)
    ax.legend(fontsize=8)


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



def plot_quality(
    leads: list[Lead] | ECGRecord,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Signal quality dashboard: SQI bar chart per lead.

    Parameters
    ----------
    leads : list[Lead] | ECGRecord
        Leads to assess.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt
    from ecgdatakit.processing.quality import signal_quality_index, snr_estimate

    lead_list, _ = _resolve_leads(leads)
    if not lead_list:
        fig, _ = plt.subplots(figsize=figsize)
        return fig

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
    ax.set_title("Signal Quality per Lead")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Excellent (>0.8)"),
        Patch(facecolor="#ff7f0e", label="Acceptable (0.5–0.8)"),
        Patch(facecolor="#d62728", label="Unacceptable (<0.5)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig



def plot_report(
    record: ECGRecord,
    figsize: tuple[float, float] = (16, 20),
) -> Figure:
    """Comprehensive ECG report page.

    Includes header with patient/device info, 12-lead grid,
    rhythm strip, and quality indicators.

    Parameters
    ----------
    record : ECGRecord
        Full ECG record.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(6, 4, figure=fig, height_ratios=[0.6, 1, 1, 1, 0.7, 0.8], hspace=0.35, wspace=0.15)

    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.axis("off")
    _draw_header(ax_hdr, record)

    leads = record.leads
    for row_idx, row_labels in enumerate(GRID_12LEAD):
        for col_idx, lbl in enumerate(row_labels):
            ax = fig.add_subplot(gs[1 + row_idx, col_idx])
            ld = _find_lead(leads, lbl)
            if ld is not None:
                t = time_axis(ld)
                max_s = int(10.0 * ld.sample_rate)
                sl = slice(0, min(max_s, len(ld.samples)))
                ax.plot(t[sl], ld.samples[sl], color=lead_color(lbl), linewidth=0.7)
                ax.set_xlim(0, 10.0)
            ax.set_title(lbl, fontsize=9, loc="left", pad=2)
            _ecg_grid(ax)
            ax.tick_params(labelsize=6)
            if row_idx < 2:
                ax.set_xticklabels([])

    ax_rhythm = fig.add_subplot(gs[4, :])
    rl = _find_lead(leads, "II")
    if rl is not None:
        t = time_axis(rl)
        ax_rhythm.plot(t, rl.samples, color=lead_color("II"), linewidth=0.7)
        ax_rhythm.set_xlim(t[0], t[-1])
    ax_rhythm.set_title("II rhythm strip", fontsize=9, loc="left", pad=2)
    _ecg_grid(ax_rhythm)
    ax_rhythm.set_xlabel("Time (s)", fontsize=8)
    ax_rhythm.tick_params(labelsize=6)

    ax_qi = fig.add_subplot(gs[5, :2])
    ax_qi.axis("off")
    _draw_quality_summary(ax_qi, leads)

    ax_interp = fig.add_subplot(gs[5, 2:])
    ax_interp.axis("off")
    _draw_interpretation(ax_interp, record)

    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def _draw_quality_summary(ax, leads: list[Lead]) -> None:
    """Draw compact quality summary text."""
    from ecgdatakit.processing.quality import classify_quality, signal_quality_index

    lines = ["Signal Quality:"]
    for ld in leads[:12]:
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
    for stmt in interp.statements:
        lines.append(f"  - {stmt}")
    if not interp.statements and not interp.severity:
        lines.append("  No interpretation available")

    ax.text(
        0.02, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
    )
