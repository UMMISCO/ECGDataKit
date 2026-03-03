# Plotting API Reference

Import: `from ecgdatakit.plotting import ...`

Static plots require: `pip install ecgdatakit[plotting]` (matplotlib >= 3.7)

Interactive plots require: `pip install ecgdatakit[plotting-interactive]` (plotly >= 5.15)

> **Numpy array support:** All plotting functions accept raw **numpy arrays** in addition to `Lead` / `ECGRecord` objects. When passing numpy arrays, provide the sample rate via `fs=`:
> ```python
> # Single-lead: 1-D array
> plot_lead(my_array, fs=500)
> plot_peaks(my_array, fs=500)
> iplot_lead(my_array, fs=500)
>
> # Multi-lead: 2-D array (n_leads × n_samples) or list of 1-D arrays
> plot_leads(signals_2d, fs=500)
> plot_12lead([lead_i, lead_ii, lead_iii, ...], fs=500)
> iplot_leads(signals_2d, fs=500)
> ```
> A `TypeError` is raised if `fs` is omitted with a numpy array. When passing `Lead` / `ECGRecord` objects, `fs` is ignored.

## Static Plots (matplotlib)

All static plot functions display the figure automatically by default (`show=True`). Pass `show=False` to suppress display and get back the `matplotlib.figure.Figure` for saving or further customization. Functions with an `ax` parameter can render into an existing axes for composability; when `ax=None`, a new figure is created.

### Lead Waveforms

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_lead">plot_lead</a>(lead, peaks=None, title=None, show_grid=True, figsize=(12,3), ax=None, *, fs=None, show=True, x_axis="time")</code></td>
      <td>Single lead waveform with optional R-peak markers. ECG-style grid with major lines every 0.2s / 0.5mV. Accepts <code>Lead</code> or numpy array. Set <code>x_axis="samples"</code> for sample indices instead of time.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_leads">plot_leads</a>(leads, peaks_dict=None, title=None, show_grid=True, figsize=(12,None), share_x=True, *, fs=None, show=True, x_axis="time")</code></td>
      <td>Multiple leads stacked vertically, shared X-axis. Accepts <code>list[Lead]</code>, <code>ECGRecord</code>, 2-D numpy array, or list of 1-D arrays. Height auto-calculated.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_12lead">plot_12lead</a>(leads, record=None, paper_speed=25, amplitude=10, rhythm_lead="II", duration=10.0, figsize=(14,10), *, fs=None, show=True, x_axis="time")</code></td>
      <td>Standard 12-lead grid (4x3) with rhythm strip. Paper-style grid. Optional header with patient info and measurements when <code>record</code> is provided. Accepts <code>list[Lead]</code>, <code>ECGRecord</code>, 2-D numpy array, or list of 1-D arrays.</td>
    </tr>
  </tbody>
</table>

```python
from ecgdatakit.plotting import plot_lead, plot_leads, plot_12lead
import numpy as np

# From a Lead object
plot_lead(lead)

# From a numpy array — fs is required
signal = np.random.randn(5000)
plot_lead(signal, fs=500)

# Use sample indices on x-axis
plot_lead(lead, x_axis="samples")

# Multi-lead from a 2-D numpy array (n_leads × n_samples)
signals = np.random.randn(3, 5000)
plot_leads(signals, fs=500)

# Multi-lead from a list of 1-D numpy arrays
plot_12lead([arr_i, arr_ii, arr_iii, ...], fs=500)

# Suppress display to save to file
fig = plot_12lead(record, show=False)
fig.savefig("ecg_grid.png", dpi=150)
```

### Annotations & Beats

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_peaks">plot_peaks</a>(lead, peaks=None, title=None, figsize=(12,3), ax=None, *, fs=None, show=True, x_axis="time")</code></td>
      <td>Lead with R-peak triangles, RR interval annotations between peaks, and heart rate in corner. Auto-detects peaks if <code>None</code>. Accepts <code>Lead</code> or numpy array.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_beats">plot_beats</a>(lead, beats=None, peaks=None, overlay=True, figsize=(8,5), ax=None, *, fs=None, show=True)</code></td>
      <td>Segmented heartbeats. <code>overlay=True</code>: all beats on same axes (semi-transparent) with average bold. <code>overlay=False</code>: waterfall/stacked. Accepts <code>Lead</code> or numpy array.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_average_beat">plot_average_beat</a>(lead, peaks=None, before=0.2, after=0.4, figsize=(6,4), ax=None, *, fs=None, show=True)</code></td>
      <td>Ensemble-averaged beat with shaded +/-1 SD region. X-axis in ms relative to R-peak. Accepts <code>Lead</code> or numpy array.</td>
    </tr>
  </tbody>
</table>

### Frequency Domain

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_spectrum">plot_spectrum</a>(lead, method="welch", figsize=(10,4), ax=None, *, fs=None, show=True)</code></td>
      <td><code>"welch"</code>: PSD in dB/Hz. <code>"fft"</code>: magnitude spectrum. Shaded ECG band (0.05-150 Hz). Accepts <code>Lead</code> or numpy array.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_spectrogram">plot_spectrogram</a>(lead, nperseg=256, figsize=(12,4), ax=None, *, fs=None, show=True)</code></td>
      <td>Time-frequency spectrogram (STFT) as a colormap. X: time, Y: frequency, color: power. Accepts <code>Lead</code> or numpy array.</td>
    </tr>
  </tbody>
</table>

### HRV

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_rr_tachogram">plot_rr_tachogram</a>(rr_ms, figsize=(10,3), ax=None, *, show=True)</code></td>
      <td>RR intervals vs. beat number with mean +/- SD reference lines. Takes a numpy array of RR intervals in ms.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_poincare">plot_poincare</a>(rr_ms, figsize=(6,6), ax=None, *, show=True)</code></td>
      <td>Poincare plot: RR(n) vs RR(n+1) with SD1/SD2 ellipse and identity line. Takes a numpy array of RR intervals in ms.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_hrv_summary">plot_hrv_summary</a>(rr_ms, figsize=(14,8), *, show=True)</code></td>
      <td>4-panel dashboard: RR tachogram, Poincare plot, frequency-domain PSD with VLF/LF/HF bands shaded, and time-domain metrics table.</td>
    </tr>
  </tbody>
</table>

### Quality

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_quality">plot_quality</a>(leads, figsize=(10,5), *, fs=None, show=True)</code></td>
      <td>Bar chart of signal quality index per lead. Color-coded: green (excellent), yellow (acceptable), red (unacceptable). SNR annotated. Accepts <code>list[Lead]</code>, <code>ECGRecord</code>, 2-D numpy array, or list of 1-D arrays.</td>
    </tr>
  </tbody>
</table>

### Full Report

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_report">plot_report</a>(record, figsize=(16,20), *, show=True)</code></td>
      <td>Comprehensive ECG report: patient header, measurements table, 12-lead grid, rhythm strip, quality summary, and interpretation statements.</td>
    </tr>
  </tbody>
</table>

```python
from ecgdatakit.plotting import plot_report

# Displays automatically
plot_report(record)

# Or suppress display to save
fig = plot_report(record, show=False)
fig.savefig("full_report.pdf")
```

## Interactive Plots (plotly)

All interactive plot functions display the figure automatically by default (`show=True`). Pass `show=False` to get the `plotly.graph_objects.Figure` without displaying. Features: zoom, pan, hover with sample-level values, range sliders.

### Lead Waveforms

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_lead">iplot_lead</a>(lead, peaks=None, title=None, height=300, *, fs=None, show=True, x_axis="time")</code></td>
      <td>Interactive single lead with rangeslider, crosshair spikes, hover showing time and amplitude. Accepts <code>Lead</code> or numpy array.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_leads">iplot_leads</a>(leads, peaks_dict=None, title=None, height=None, *, fs=None, show=True, x_axis="time")</code></td>
      <td>Stacked leads with synchronized X-axis zoom. Toggle visibility via legend. Accepts <code>list[Lead]</code>, <code>ECGRecord</code>, 2-D numpy array, or list of 1-D arrays.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_12lead">iplot_12lead</a>(leads, record=None, duration=10.0, height=800, *, fs=None, show=True, x_axis="time")</code></td>
      <td>Interactive 4x3 grid + rhythm strip. Header annotation when record provided. Accepts <code>list[Lead]</code>, <code>ECGRecord</code>, 2-D numpy array, or list of 1-D arrays.</td>
    </tr>
  </tbody>
</table>

```python
from ecgdatakit.plotting import iplot_lead, iplot_leads, iplot_12lead
import numpy as np

# From a Lead object — displays automatically
iplot_lead(filtered, peaks=peaks)

# From a numpy array
signal = np.random.randn(5000)
iplot_lead(signal, fs=500)

# Multi-lead from a 2-D numpy array
signals = np.random.randn(12, 5000)
iplot_leads(signals, fs=500)

# Multi-lead from a list of 1-D arrays
iplot_12lead([arr_i, arr_ii, arr_iii, ...], fs=500)

# Suppress display
fig = iplot_12lead(record, show=False)
fig.write_html("ecg_12lead.html")
```

### Annotations

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_peaks">iplot_peaks</a>(lead, peaks=None, title=None, height=350, *, fs=None, show=True, x_axis="time")</code></td>
      <td>Lead with R-peak markers. Hover shows peak index, RR interval, and instantaneous HR. Accepts <code>Lead</code> or numpy array.</td>
    </tr>
  </tbody>
</table>

### Frequency Domain

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_spectrum">iplot_spectrum</a>(lead, method="welch", height=400, *, fs=None, show=True)</code></td>
      <td>Interactive spectrum with hover showing frequency and power. Shaded ECG band. Accepts <code>Lead</code> or numpy array.</td>
    </tr>
  </tbody>
</table>

### HRV

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_rr_tachogram">iplot_rr_tachogram</a>(rr_ms, height=300, *, show=True)</code></td>
      <td>Interactive RR tachogram with mean +/- SD lines. Hover shows beat number and RR value.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_poincare">iplot_poincare</a>(rr_ms, height=500, *, show=True)</code></td>
      <td>Interactive Poincare with SD1/SD2 ellipse. Hover shows beat pair indices and RR values.</td>
    </tr>
  </tbody>
</table>

### Full Report

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_report">iplot_report</a>(record, height=1200, *, show=True, x_axis="time")</code></td>
      <td>Full interactive report with all leads, rhythm strip with rangeslider, and header annotation.</td>
    </tr>
  </tbody>
</table>