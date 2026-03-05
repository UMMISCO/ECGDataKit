# Static Plots (matplotlib)

All static plot functions display the figure automatically by default (`show=True`). Pass `show=False` to suppress display and get back the `matplotlib.figure.Figure` for saving or further customization. Functions with an `ax` parameter can render into an existing axes for composability; when `ax=None`, a new figure is created.

```{eval-rst}
.. currentmodule:: ecgdatakit.plotting
```

## Lead Waveforms

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_lead` | Plot a single ECG lead waveform |
| {func}`~ecgdatakit.plotting.plot_leads` | Plot multiple leads in a grid layout |
| {func}`~ecgdatakit.plotting.plot_12lead` | Plot 12 leads with standard lead names (I, II, III, aVR, …, V6) |

```{eval-rst}
.. autofunction:: plot_lead
.. autofunction:: plot_leads
.. autofunction:: plot_12lead
```

## Annotations & Beats

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_peaks` | Plot lead with R-peak markers and RR interval annotations |
| {func}`~ecgdatakit.plotting.plot_beats` | Plot segmented heartbeats |
| {func}`~ecgdatakit.plotting.plot_average_beat` | Plot ensemble-averaged beat with ±1 SD shading |

```{eval-rst}
.. autofunction:: plot_peaks
.. autofunction:: plot_beats
.. autofunction:: plot_average_beat
```

## Frequency Domain

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_spectrum` | Plot power spectral density or FFT magnitude spectrum |
| {func}`~ecgdatakit.plotting.plot_spectrogram` | Plot time-frequency spectrogram (STFT) |

```{eval-rst}
.. autofunction:: plot_spectrum
.. autofunction:: plot_spectrogram
```

## HRV

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_rr_tachogram` | Plot RR interval tachogram |
| {func}`~ecgdatakit.plotting.plot_poincare` | Poincaré plot: RR(n) vs RR(n+1) with SD1/SD2 ellipse |
| {func}`~ecgdatakit.plotting.plot_hrv_summary` | Combined HRV dashboard: tachogram, Poincaré, frequency bands, metrics |

```{eval-rst}
.. autofunction:: plot_rr_tachogram
.. autofunction:: plot_poincare
.. autofunction:: plot_hrv_summary
```

## Quality

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_quality` | Signal quality dashboard: SQI bar chart per lead |

```{eval-rst}
.. autofunction:: plot_quality
```

## Full Report

| | |
|---|---|
| {func}`~ecgdatakit.plotting.plot_report` | Comprehensive ECG report page |

```{eval-rst}
.. autofunction:: plot_report
```
