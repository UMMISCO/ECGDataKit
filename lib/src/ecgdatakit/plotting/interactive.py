"""Interactive ECG plots using plotly.

All public functions return ``plotly.graph_objects.Figure``.

Requires: ``pip install ecgdatakit[plotting-interactive]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import ECGRecord, Lead
from ecgdatakit.plotting._core import (
    GRID_12LEAD,
    _find_lead,
    _resolve_leads,
    lead_color,
    require_plotly,
    time_axis,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go



def iplot_lead(
    lead: Lead,
    peaks: NDArray[np.intp] | None = None,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Interactive single lead with hover showing time/amplitude.

    Parameters
    ----------
    lead : Lead
        ECG lead to plot.
    peaks : NDArray | None
        Optional R-peak indices to mark.
    title : str | None
        Figure title.
    height : int
        Figure height in pixels.
    """
    require_plotly()
    import plotly.graph_objects as go

    t = time_axis(lead)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=lead.samples,
        mode="lines",
        name=lead.label,
        line=dict(color=lead_color(lead.label), width=1),
        hovertemplate="Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>",
    ))

    if peaks is not None and len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=t[peaks], y=lead.samples[peaks],
            mode="markers",
            name="R-peaks",
            marker=dict(color="red", size=8, symbol="triangle-down"),
            hovertemplate="Peak at %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=title or lead.label,
        xaxis_title="Time (s)",
        yaxis_title=f"Amplitude ({lead.units})" if lead.units else "Amplitude",
        height=height,
        xaxis=dict(
            rangeslider=dict(visible=True),
            showspikes=True, spikemode="across", spikethickness=1,
        ),
        yaxis=dict(showspikes=True, spikemode="across", spikethickness=1),
        hovermode="x unified",
    )
    return fig


def iplot_leads(
    leads: list[Lead] | ECGRecord,
    peaks_dict: dict[str, NDArray[np.intp]] | None = None,
    title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Interactive stacked leads with synchronized X-axis zoom.

    Parameters
    ----------
    leads : list[Lead] | ECGRecord
        Leads to plot.
    peaks_dict : dict | None
        ``{label: peaks_array}`` for per-lead peak markers.
    title : str | None
        Overall title.
    height : int | None
        Figure height (auto-calculated if ``None``).
    """
    require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    lead_list, _ = _resolve_leads(leads)
    n = len(lead_list)
    if n == 0:
        return go.Figure()

    h = height or max(300, 150 * n)
    subplot_titles = [ld.label for ld in lead_list]

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
    )

    for i, ld in enumerate(lead_list, start=1):
        t = time_axis(ld)
        fig.add_trace(
            go.Scatter(
                x=t, y=ld.samples,
                mode="lines",
                name=ld.label,
                line=dict(color=lead_color(ld.label), width=1),
                hovertemplate="%{y:.3f}<extra></extra>",
            ),
            row=i, col=1,
        )

        if peaks_dict and ld.label in peaks_dict:
            pk = peaks_dict[ld.label]
            fig.add_trace(
                go.Scatter(
                    x=t[pk], y=ld.samples[pk],
                    mode="markers",
                    name=f"{ld.label} peaks",
                    marker=dict(color="red", size=6, symbol="triangle-down"),
                    showlegend=False,
                ),
                row=i, col=1,
            )

    fig.update_layout(
        title=title or "ECG Leads",
        height=h,
        hovermode="x unified",
        showlegend=True,
    )
    fig.update_xaxes(title_text="Time (s)", row=n, col=1)
    return fig


def iplot_12lead(
    leads: list[Lead] | ECGRecord,
    record: ECGRecord | None = None,
    duration: float = 10.0,
    height: int = 800,
) -> go.Figure:
    """Interactive 12-lead grid.

    Parameters
    ----------
    leads : list[Lead] | ECGRecord
        Leads (or record) to plot.
    record : ECGRecord | None
        Optional record for header annotations.
    duration : float
        Seconds per cell (default 10).
    height : int
        Figure height in pixels.
    """
    require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    lead_list, rec = _resolve_leads(leads)
    if record is not None:
        rec = record

    subplot_titles = []
    for row_labels in GRID_12LEAD:
        subplot_titles.extend(row_labels)
    subplot_titles.append("II Rhythm Strip")
    while len(subplot_titles) < 16:
        subplot_titles.append("")

    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=subplot_titles[:16],
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 4}, None, None, None],
        ],
    )

    for row_idx, row_labels in enumerate(GRID_12LEAD):
        for col_idx, lbl in enumerate(row_labels):
            ld = _find_lead(lead_list, lbl)
            if ld is not None:
                t = time_axis(ld)
                max_s = int(duration * ld.sample_rate)
                sl = slice(0, min(max_s, len(ld.samples)))
                fig.add_trace(
                    go.Scatter(
                        x=t[sl], y=ld.samples[sl],
                        mode="lines",
                        name=lbl,
                        line=dict(color=lead_color(lbl), width=1),
                        showlegend=False,
                        hovertemplate=f"{lbl}<br>%{{x:.3f}}s: %{{y:.3f}}<extra></extra>",
                    ),
                    row=row_idx + 1, col=col_idx + 1,
                )

    rl = _find_lead(lead_list, "II")
    if rl is not None:
        t = time_axis(rl)
        fig.add_trace(
            go.Scatter(
                x=t, y=rl.samples,
                mode="lines",
                name="II rhythm",
                line=dict(color=lead_color("II"), width=1),
                showlegend=False,
                hovertemplate="II<br>%{x:.3f}s: %{y:.3f}<extra></extra>",
            ),
            row=4, col=1,
        )

    if rec is not None:
        header_text = _build_header_text(rec)
        fig.add_annotation(
            text=header_text,
            xref="paper", yref="paper",
            x=0, y=1.08,
            showarrow=False,
            font=dict(size=10, family="monospace"),
            align="left",
        )

    fig.update_layout(
        height=height,
        hovermode="closest",
        margin=dict(t=80 if rec else 40),
    )
    return fig


def _build_header_text(record: ECGRecord) -> str:
    """Build compact header text for plotly annotations."""
    parts = []
    p = record.patient
    name = f"{p.first_name} {p.last_name}".strip()
    if name:
        parts.append(f"<b>{name}</b>")
    if p.patient_id:
        parts.append(f"ID: {p.patient_id}")
    if p.age is not None:
        parts.append(f"Age: {p.age}")
    if p.sex:
        parts.append(f"Sex: {p.sex}")

    m = record.measurements
    meas = []
    if m.heart_rate is not None:
        meas.append(f"HR:{m.heart_rate}")
    if m.pr_interval is not None:
        meas.append(f"PR:{m.pr_interval}")
    if m.qrs_duration is not None:
        meas.append(f"QRS:{m.qrs_duration}")
    if m.qtc_bazett is not None:
        meas.append(f"QTc:{m.qtc_bazett}")
    if meas:
        parts.append(" | ".join(meas))

    return "  —  ".join(parts) if parts else "ECG Report"



def iplot_peaks(
    lead: Lead,
    peaks: NDArray[np.intp] | None = None,
    title: str | None = None,
    height: int = 350,
) -> go.Figure:
    """Interactive lead with R-peak markers.

    Hover on peaks shows peak index, RR interval, and instantaneous HR.

    Parameters
    ----------
    lead : Lead
        ECG lead to plot.
    peaks : NDArray | None
        R-peak indices. Auto-detected if ``None``.
    """
    from ecgdatakit.processing.peaks import detect_r_peaks

    require_plotly()
    import plotly.graph_objects as go

    if peaks is None:
        peaks = detect_r_peaks(lead)

    t = time_axis(lead)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=lead.samples,
        mode="lines",
        name=lead.label,
        line=dict(color=lead_color(lead.label), width=1),
        hovertemplate="Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>",
    ))

    if len(peaks) > 0:
        hover_texts = []
        for i, p in enumerate(peaks):
            parts = [f"Peak #{i}", f"Time: {t[p]:.3f}s"]
            if i > 0:
                rr = (peaks[i] - peaks[i - 1]) / lead.sample_rate * 1000
                hr = 60_000.0 / rr if rr > 0 else 0
                parts.append(f"RR: {rr:.0f}ms")
                parts.append(f"HR: {hr:.0f}bpm")
            hover_texts.append("<br>".join(parts))

        fig.add_trace(go.Scatter(
            x=t[peaks], y=lead.samples[peaks],
            mode="markers",
            name="R-peaks",
            marker=dict(color="red", size=9, symbol="triangle-down"),
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    fig.update_layout(
        title=title or f"{lead.label} — R-peaks",
        xaxis_title="Time (s)",
        yaxis_title=f"Amplitude ({lead.units})" if lead.units else "Amplitude",
        height=height,
        xaxis=dict(rangeslider=dict(visible=True)),
        hovermode="closest",
    )
    return fig



def iplot_spectrum(
    lead: Lead,
    method: str = "welch",
    height: int = 400,
) -> go.Figure:
    """Interactive spectrum with frequency band highlighting.

    Parameters
    ----------
    lead : Lead
        ECG lead.
    method : str
        ``"welch"`` for PSD or ``"fft"`` for magnitude spectrum.
    """
    from ecgdatakit.processing.transforms import fft as ecg_fft
    from ecgdatakit.processing.transforms import power_spectrum

    require_plotly()
    import plotly.graph_objects as go

    fig = go.Figure()

    if method == "welch":
        freqs, power = power_spectrum(lead)
        power_db = 10 * np.log10(np.maximum(power, 1e-20))
        fig.add_trace(go.Scatter(
            x=freqs, y=power_db,
            mode="lines",
            name="PSD",
            line=dict(color=lead_color(lead.label), width=1),
            hovertemplate="Freq: %{x:.2f} Hz<br>Power: %{y:.1f} dB/Hz<extra></extra>",
        ))
        y_label = "Power (dB/Hz)"
        title_suffix = "PSD (Welch)"
    else:
        freqs, mags = ecg_fft(lead)
        fig.add_trace(go.Scatter(
            x=freqs, y=mags,
            mode="lines",
            name="FFT",
            line=dict(color=lead_color(lead.label), width=1),
            hovertemplate="Freq: %{x:.2f} Hz<br>Magnitude: %{y:.4f}<extra></extra>",
        ))
        y_label = "Magnitude"
        title_suffix = "FFT Magnitude"

    nyquist = lead.sample_rate / 2
    fig.add_vrect(x0=0.05, x1=150, fillcolor="green", opacity=0.05, line_width=0, annotation_text="ECG band")

    fig.update_layout(
        title=f"{lead.label} — {title_suffix}",
        xaxis_title="Frequency (Hz)",
        yaxis_title=y_label,
        height=height,
        xaxis=dict(range=[0, min(nyquist, 250)]),
        hovermode="x unified",
    )
    return fig



def iplot_rr_tachogram(
    rr_ms: NDArray[np.float64],
    height: int = 300,
) -> go.Figure:
    """Interactive RR interval tachogram.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.
    """
    require_plotly()
    import plotly.graph_objects as go

    beats = np.arange(len(rr_ms))
    mean_rr = float(rr_ms.mean())
    std_rr = float(rr_ms.std())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=beats, y=rr_ms,
        mode="lines+markers",
        name="RR intervals",
        line=dict(color="#1f77b4", width=1),
        marker=dict(size=3),
        hovertemplate="Beat #%{x}<br>RR: %{y:.0f} ms<extra></extra>",
    ))

    fig.add_hline(y=mean_rr, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_rr:.0f} ms")
    fig.add_hline(y=mean_rr + std_rr, line_dash="dot", line_color="orange",
                  annotation_text=f"+SD: {std_rr:.0f} ms")
    fig.add_hline(y=mean_rr - std_rr, line_dash="dot", line_color="orange")

    fig.update_layout(
        title="RR Tachogram",
        xaxis_title="Beat number",
        yaxis_title="RR interval (ms)",
        height=height,
        hovermode="x unified",
    )
    return fig


def iplot_poincare(
    rr_ms: NDArray[np.float64],
    height: int = 500,
) -> go.Figure:
    """Interactive Poincaré plot with SD1/SD2 ellipse.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.
    """
    require_plotly()
    import plotly.graph_objects as go

    fig = go.Figure()

    if len(rr_ms) < 2:
        fig.add_annotation(text="Need >= 2 RR intervals", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    x = rr_ms[:-1]
    y = rr_ms[1:]

    hover = [f"Beat {i}→{i+1}<br>RR(n): {x[i]:.0f} ms<br>RR(n+1): {y[i]:.0f} ms"
             for i in range(len(x))]

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        name="RR pairs",
        marker=dict(color="#1f77b4", size=5, opacity=0.5),
        hovertext=hover,
        hoverinfo="text",
    ))

    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    margin = (hi - lo) * 0.1
    fig.add_trace(go.Scatter(
        x=[lo - margin, hi + margin],
        y=[lo - margin, hi + margin],
        mode="lines",
        name="Identity",
        line=dict(color="black", width=0.5, dash="dash"),
        showlegend=False,
    ))

    sd1 = float(np.std(y - x, ddof=1) / np.sqrt(2))
    sd2 = float(np.std(y + x, ddof=1) / np.sqrt(2))
    cx, cy = float(x.mean()), float(y.mean())

    theta = np.linspace(0, 2 * np.pi, 100)
    cos45, sin45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
    ex = sd2 * np.cos(theta)
    ey = sd1 * np.sin(theta)
    rx = cx + ex * cos45 - ey * sin45
    ry = cy + ex * sin45 + ey * cos45

    fig.add_trace(go.Scatter(
        x=rx, y=ry,
        mode="lines",
        name=f"SD1={sd1:.1f}, SD2={sd2:.1f}",
        line=dict(color="red", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        title="Poincaré Plot",
        xaxis_title="RR(n) (ms)",
        yaxis_title="RR(n+1) (ms)",
        height=height,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        hovermode="closest",
    )
    return fig



def iplot_report(
    record: ECGRecord,
    height: int = 1200,
) -> go.Figure:
    """Interactive full ECG report.

    Parameters
    ----------
    record : ECGRecord
        Full ECG record.
    """
    require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    leads = record.leads
    n_leads = min(len(leads), 12)

    total_rows = n_leads + 1
    subplot_titles = [ld.label for ld in leads[:n_leads]] + ["II Rhythm Strip"]

    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.015,
    )

    for i, ld in enumerate(leads[:n_leads], start=1):
        t = time_axis(ld)
        max_s = int(10.0 * ld.sample_rate)
        sl = slice(0, min(max_s, len(ld.samples)))
        fig.add_trace(
            go.Scatter(
                x=t[sl], y=ld.samples[sl],
                mode="lines",
                name=ld.label,
                line=dict(color=lead_color(ld.label), width=1),
                showlegend=False,
                hovertemplate=f"{ld.label}<br>%{{x:.3f}}s: %{{y:.3f}}<extra></extra>",
            ),
            row=i, col=1,
        )

    rl = _find_lead(leads, "II")
    if rl is not None:
        t = time_axis(rl)
        fig.add_trace(
            go.Scatter(
                x=t, y=rl.samples,
                mode="lines",
                name="II rhythm",
                line=dict(color=lead_color("II"), width=1),
                showlegend=False,
                hovertemplate="II<br>%{x:.3f}s: %{y:.3f}<extra></extra>",
            ),
            row=total_rows, col=1,
        )
        fig.update_xaxes(rangeslider=dict(visible=True), row=total_rows, col=1)

    header_text = _build_header_text(record)
    fig.add_annotation(
        text=header_text,
        xref="paper", yref="paper",
        x=0, y=1.02,
        showarrow=False,
        font=dict(size=11, family="monospace"),
        align="left",
    )

    fig.update_layout(
        height=height,
        hovermode="closest",
        margin=dict(t=60),
    )
    fig.update_xaxes(title_text="Time (s)", row=total_rows, col=1)
    return fig
