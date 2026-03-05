# Plotting API Reference

Import: `from ecgdatakit.plotting import ...`

Static plots require: `pip install ecgdatakit[plotting]` (matplotlib >= 3.7)

Interactive plots require: `pip install ecgdatakit[plotting-interactive]` (plotly >= 5.15)

```{note}
All plotting functions accept raw **numpy arrays** in addition to `Lead` / `ECGRecord` objects. When passing numpy arrays, provide the sample rate via `fs`:

    # Single-lead: 1-D array
    plot_lead(my_array, fs=500)

    # Multi-lead: 2-D array (n_leads × n_samples)
    plot_leads(signals_2d, fs=500)

A `TypeError` is raised if `fs` is omitted with a numpy array. When passing `Lead` / `ECGRecord` objects, `fs` is ignored.
```

## {doc}`Static Plots (matplotlib) <plotting/static>`

### Lead Waveforms

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_lead` | Plot a single ECG lead waveform |
| {func}`~ecgdatakit.plotting.plot_leads` | Plot multiple leads in a grid layout |
| {func}`~ecgdatakit.plotting.plot_12lead` | Plot 12 leads with standard lead names (I, II, III, aVR, …, V6) |

### Annotations & Beats

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_peaks` | Plot lead with R-peak markers and RR interval annotations |
| {func}`~ecgdatakit.plotting.plot_beats` | Plot segmented heartbeats |
| {func}`~ecgdatakit.plotting.plot_average_beat` | Plot ensemble-averaged beat with ±1 SD shading |

### Frequency Domain

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_spectrum` | Plot power spectral density or FFT magnitude spectrum |
| {func}`~ecgdatakit.plotting.plot_spectrogram` | Plot time-frequency spectrogram (STFT) |

### HRV

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_rr_tachogram` | Plot RR interval tachogram |
| {func}`~ecgdatakit.plotting.plot_poincare` | Poincaré plot: RR(n) vs RR(n+1) with SD1/SD2 ellipse |
| {func}`~ecgdatakit.plotting.plot_hrv_summary` | Combined HRV dashboard: tachogram, Poincaré, frequency bands, metrics |

### Quality & Report

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_quality` | Signal quality dashboard: SQI bar chart per lead |
| {func}`~ecgdatakit.plotting.plot_report` | Comprehensive ECG report page |

## {doc}`Interactive Plots (plotly) <plotting/interactive>`

### Lead Waveforms

| | |
|---|---|
| {func}`~ecgdatakit.plotting.iplot_lead` | Interactive single lead with hover showing time/amplitude |
| {func}`~ecgdatakit.plotting.iplot_leads` | Interactive leads in a grid layout |
| {func}`~ecgdatakit.plotting.iplot_12lead` | Interactive 12-lead plot with standard lead names |

### Annotations & Frequency Domain

| | |
|---|---|
| {func}`~ecgdatakit.plotting.iplot_peaks` | Interactive lead with R-peak markers |
| {func}`~ecgdatakit.plotting.iplot_spectrum` | Interactive spectrum with frequency band highlighting |

### HRV & Report

| | |
|---|---|
| {func}`~ecgdatakit.plotting.iplot_rr_tachogram` | Interactive RR interval tachogram |
| {func}`~ecgdatakit.plotting.iplot_poincare` | Interactive Poincaré plot with SD1/SD2 ellipse |
| {func}`~ecgdatakit.plotting.iplot_report` | Interactive full ECG report |

```{toctree}
:hidden:

plotting/static
plotting/interactive
```
