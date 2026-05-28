"""ECG visualization tools.

Requires: ``pip install ecgdatakit[plotting]`` for static plots (matplotlib)
          ``pip install ecgdatakit[plotting-interactive]`` for interactive plots (plotly)

Modules
-------
static
    matplotlib-based static plots: single lead, multi-lead, 12-lead grid,
    R-peak annotations, beat segmentation, spectrum, spectrogram, HRV
    dashboard, quality dashboard, full report.
interactive
    plotly-based interactive plots with zoom, pan, hover, and range sliders.
"""

from ecgdatakit.plotting.static import (
    plot_12lead,
    plot_average_beat,
    plot_beats,
    plot_hrv_summary,
    plot_lead,
    plot_leads,
    plot_peaks,
    plot_poincare,
    plot_quality,
    plot_report,
    plot_rr_tachogram,
    plot_spectrogram,
    plot_spectrum,
)
from ecgdatakit.plotting.interactive import (
    iplot_12lead,
    iplot_lead,
    iplot_leads,
    iplot_peaks,
    iplot_poincare,
    iplot_report,
    iplot_rr_tachogram,
    iplot_spectrum,
)

__all__ = [
    "plot_lead",
    "plot_leads",
    "plot_12lead",
    "plot_peaks",
    "plot_beats",
    "plot_average_beat",
    "plot_spectrum",
    "plot_spectrogram",
    "plot_rr_tachogram",
    "plot_poincare",
    "plot_hrv_summary",
    "plot_quality",
    "plot_report",
    "iplot_lead",
    "iplot_leads",
    "iplot_12lead",
    "iplot_peaks",
    "iplot_spectrum",
    "iplot_rr_tachogram",
    "iplot_poincare",
    "iplot_report",
]
