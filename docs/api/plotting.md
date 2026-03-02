# Plotting API Reference

Import: `from ecgdatakit.plotting import ...`

Static plots require: `pip install ecgdatakit[plotting]` (matplotlib ≥ 3.7)

Interactive plots require: `pip install ecgdatakit[plotting-interactive]` (plotly ≥ 5.15)

> **Note:** All single-lead plotting functions accept both `Lead` objects and raw numpy arrays. When passing a numpy array, provide the sample rate via `fs`:
> ```python
> plot_lead(my_array, fs=500)
> ```
> See [Data Models](parsing.md#working-with-data-models) for details.

## Static Plots (matplotlib)

All functions return `matplotlib.figure.Figure`. Functions with an `ax` parameter can render into an existing axes for composability. When `ax=None`, a new figure is created.

### Lead Waveforms

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_lead">plot_lead</a>(lead, peaks=None, title=None, show_grid=True, figsize=(12,3), ax=None)</code></td>
      <td>Single lead waveform with optional R-peak markers. ECG-style grid with major lines every 0.2s / 0.5mV.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_leads">plot_leads</a>(leads, peaks_dict=None, title=None, show_grid=True, figsize=(12,None), share_x=True)</code></td>
      <td>Multiple leads stacked vertically, shared X-axis. Accepts <code>list[Lead]</code> or <code>ECGRecord</code>. Height auto-calculated.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_12lead">plot_12lead</a>(leads, record=None, paper_speed=25, amplitude=10, rhythm_lead="II", duration=10.0, figsize=(14,10))</code></td>
      <td>Standard 12-lead grid (4×3) with rhythm strip. Paper-style grid. Optional header with patient info and measurements when <code>record</code> is provided. Accepts <code>list[Lead]</code> or <code>ECGRecord</code>.</td>
    </tr>
  </tbody>
</table>

```python
from ecgdatakit.plotting import plot_12lead

# From an ECGRecord (header with patient info)
fig = plot_12lead(record)
fig.savefig("ecg_grid.png", dpi=150)

# From a list of leads (no header)
fig = plot_12lead(record.leads)
```

### Annotations & Beats

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_peaks">plot_peaks</a>(lead, peaks=None, title=None, figsize=(12,3), ax=None)</code></td>
      <td>Lead with R-peak triangles, RR interval annotations between peaks, and heart rate in corner. Auto-detects peaks if <code>None</code>.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_beats">plot_beats</a>(lead, beats=None, peaks=None, overlay=True, figsize=(8,5), ax=None)</code></td>
      <td>Segmented heartbeats. <code>overlay=True</code>: all beats on same axes (semi-transparent) with average bold. <code>overlay=False</code>: waterfall/stacked.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_average_beat">plot_average_beat</a>(lead, peaks=None, before=0.2, after=0.4, figsize=(6,4), ax=None)</code></td>
      <td>Ensemble-averaged beat with shaded ±1 SD region. X-axis in ms relative to R-peak.</td>
    </tr>
  </tbody>
</table>

### Frequency Domain

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_spectrum">plot_spectrum</a>(lead, method="welch", figsize=(10,4), ax=None)</code></td>
      <td><code>"welch"</code>: PSD in dB/Hz. <code>"fft"</code>: magnitude spectrum. Shaded ECG band (0.05–150 Hz).</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_spectrogram">plot_spectrogram</a>(lead, nperseg=256, figsize=(12,4), ax=None)</code></td>
      <td>Time-frequency spectrogram (STFT) as a colormap. X: time, Y: frequency, color: power.</td>
    </tr>
  </tbody>
</table>

### HRV

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_rr_tachogram">plot_rr_tachogram</a>(rr_ms, figsize=(10,3), ax=None)</code></td>
      <td>RR intervals vs. beat number with mean ± SD reference lines.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_poincare">plot_poincare</a>(rr_ms, figsize=(6,6), ax=None)</code></td>
      <td>Poincaré plot: RR(n) vs RR(n+1) with SD1/SD2 ellipse and identity line.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_hrv_summary">plot_hrv_summary</a>(rr_ms, figsize=(14,8))</code></td>
      <td>4-panel dashboard: RR tachogram, Poincaré plot, frequency-domain PSD with VLF/LF/HF bands shaded, and time-domain metrics table.</td>
    </tr>
  </tbody>
</table>

### Quality

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_quality">plot_quality</a>(leads, figsize=(10,5))</code></td>
      <td>Bar chart of signal quality index per lead. Color-coded: green (excellent), yellow (acceptable), red (unacceptable). SNR annotated.</td>
    </tr>
  </tbody>
</table>

### Full Report

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.plot_report">plot_report</a>(record, figsize=(16,20))</code></td>
      <td>Comprehensive ECG report: patient header, measurements table, 12-lead grid, rhythm strip, quality summary, and interpretation statements.</td>
    </tr>
  </tbody>
</table>

```python
from ecgdatakit.plotting import plot_report

fig = plot_report(record)
fig.savefig("full_report.pdf")
```

## Interactive Plots (plotly)

All functions return `plotly.graph_objects.Figure`. Features: zoom, pan, hover with sample-level values, range sliders.

### Lead Waveforms

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_lead">iplot_lead</a>(lead, peaks=None, title=None, height=300)</code></td>
      <td>Interactive single lead with rangeslider, crosshair spikes, hover showing time and amplitude.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_leads">iplot_leads</a>(leads, peaks_dict=None, title=None, height=None)</code></td>
      <td>Stacked leads with synchronized X-axis zoom. Toggle visibility via legend. Accepts <code>list[Lead]</code> or <code>ECGRecord</code>.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_12lead">iplot_12lead</a>(leads, record=None, duration=10.0, height=800)</code></td>
      <td>Interactive 4×3 grid + rhythm strip. Header annotation when record provided.</td>
    </tr>
  </tbody>
</table>

```python
from ecgdatakit.plotting import iplot_lead, iplot_12lead

# Single lead with peaks
fig = iplot_lead(filtered, peaks=peaks)
fig.show()

# Full 12-lead interactive
fig = iplot_12lead(record)
fig.show()
```

### Annotations

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_peaks">iplot_peaks</a>(lead, peaks=None, title=None, height=350)</code></td>
      <td>Lead with R-peak markers. Hover shows peak index, RR interval, and instantaneous HR.</td>
    </tr>
  </tbody>
</table>

### Frequency Domain

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_spectrum">iplot_spectrum</a>(lead, method="welch", height=400)</code></td>
      <td>Interactive spectrum with hover showing frequency and power. Shaded ECG band.</td>
    </tr>
  </tbody>
</table>

### HRV

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_rr_tachogram">iplot_rr_tachogram</a>(rr_ms, height=300)</code></td>
      <td>Interactive RR tachogram with mean ± SD lines. Hover shows beat number and RR value.</td>
    </tr>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_poincare">iplot_poincare</a>(rr_ms, height=500)</code></td>
      <td>Interactive Poincaré with SD1/SD2 ellipse. Hover shows beat pair indices and RR values.</td>
    </tr>
  </tbody>
</table>

### Full Report

<table>
  <thead><tr><th>Function</th><th>Description</th></tr></thead>
  <tbody>
    <tr>
      <td><code><a href="reference.html#ecgdatakit.plotting.iplot_report">iplot_report</a>(record, height=1200)</code></td>
      <td>Full interactive report with all leads, rhythm strip with rangeslider, and header annotation.</td>
    </tr>
  </tbody>
</table>
