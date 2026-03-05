# Interactive Plots (plotly)

All interactive plot functions display the figure automatically by default (`show=True`). Pass `show=False` to get the `plotly.graph_objects.Figure` without displaying. Features: zoom, pan, hover with sample-level values, range sliders.

```{eval-rst}
.. currentmodule:: ecgdatakit.plotting
```

## Lead Waveforms

| | |
|---|---|
| {func}`~ecgdatakit.plotting.iplot_lead` | Interactive single lead with hover showing time/amplitude |
| {func}`~ecgdatakit.plotting.iplot_leads` | Interactive leads in a grid layout |
| {func}`~ecgdatakit.plotting.iplot_12lead` | Interactive 12-lead plot with standard lead names |

```{eval-rst}
.. autofunction:: iplot_lead
.. autofunction:: iplot_leads
.. autofunction:: iplot_12lead
```

## Annotations

| | |
|---|---|
| {func}`~ecgdatakit.plotting.iplot_peaks` | Interactive lead with R-peak markers |

```{eval-rst}
.. autofunction:: iplot_peaks
```

## Frequency Domain

| | |
|---|---|
| {func}`~ecgdatakit.plotting.iplot_spectrum` | Interactive spectrum with frequency band highlighting |

```{eval-rst}
.. autofunction:: iplot_spectrum
```

## HRV

| | |
|---|---|
| {func}`~ecgdatakit.plotting.iplot_rr_tachogram` | Interactive RR interval tachogram |
| {func}`~ecgdatakit.plotting.iplot_poincare` | Interactive Poincaré plot with SD1/SD2 ellipse |

```{eval-rst}
.. autofunction:: iplot_rr_tachogram
.. autofunction:: iplot_poincare
```

## Full Report

| | |
|---|---|
| {func}`~ecgdatakit.plotting.iplot_report` | Interactive full ECG report |

```{eval-rst}
.. autofunction:: iplot_report
```
