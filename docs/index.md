---
hide-toc: true
---

# ECGDataKit

```{raw} html
<!-- Hide sidebar on landing page -->
<style>
  .sidebar-drawer { display: none !important; }
  .sidebar-toggle { display: none !important; }
  label[for="__navigation"] { display: none !important; }
  .page { --sidebar-width: 0px !important; }
  .main .content { max-width: 900px; margin: 0 auto; }
  article h1 { display: none; }
</style>

<!-- ── Hero Section ──────────────────────────────────── -->
<div class="landing-hero">
  <div class="landing-hero-logo">
    <img src="_static/logo.svg" alt="ECGDataKit">
  </div>
  <h2 class="landing-hero-title"><span class="lh-ecg">ECG</span><span class="lh-dk">DataKit</span></h2>
  <span class="landing-hero-version">v0.0.6</span>
  <p class="landing-hero-subtitle">A Python library for parsing, processing, and visualizing multi-format ECG files.</p>

  <div class="landing-pills">
    <a href="guides/formats.html" class="landing-pill">
      <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
      12 ECG Formats
    </a>
    <a href="api/processing.html" class="landing-pill">
      <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
      Signal Processing
    </a>
    <a href="api/plotting.html" class="landing-pill">
      <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z"/><path d="M4 9h16M8 4v16"/></svg>
      ECG Plotting
    </a>
  </div>

  <div class="landing-cta">
    <a href="guides/getting-started.html" class="landing-btn landing-btn-primary">Get Started &rarr;</a>
    <a href="api/parsing.html" class="landing-btn landing-btn-outline">API Reference</a>
  </div>

  <div class="landing-install">
    <code>$ pip install ecgdatakit</code>
  </div>
</div>

<!-- ── Feature Cards ─────────────────────────────────── -->
<div class="landing-features">
  <div class="landing-card">
    <div class="landing-card-icon landing-card-icon-parse">
      <svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
    </div>
    <h3>Multi-Format Parsing</h3>
    <p>Parse HL7 aECG, Philips Sierra, GE MUSE, SCP-ECG, DICOM, EDF, WFDB, MFER, and more into one unified data structure.</p>
  </div>
  <div class="landing-card">
    <div class="landing-card-icon landing-card-icon-proc">
      <svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
    </div>
    <h3>Signal Processing</h3>
    <p>Butterworth filters, R-peak detection (Pan-Tompkins &amp; Shannon energy), HRV analysis, FFT, signal quality, lead derivation, ECG cleaning, and DeepFADE neural-net denoising.</p>
  </div>
  <div class="landing-card">
    <div class="landing-card-icon landing-card-icon-plot">
      <svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z"/><path d="M4 9h16M8 4v16"/></svg>
    </div>
    <h3>Visualization</h3>
    <p>Standard 12-lead grids, R-peak annotations, HRV dashboards, spectrograms, and full ECG reports. Static or interactive.</p>
  </div>
</div>

<!-- ── Quick Example ──────────────────────────────────── -->
<h2 class="landing-section-title">Quick Example</h2>
```

```python
from ecgdatakit import FileParser
from ecgdatakit.processing import diagnostic_filter, detect_r_peaks, heart_rate
from ecgdatakit.plotting import plot_12lead, plot_peaks

# Parse any ECG file (auto-detect format)
record = FileParser().parse("ecg_file.xml")

# Filter and detect R-peaks
lead = record.leads[1]  # Lead II
filtered = diagnostic_filter(lead)
peaks = detect_r_peaks(filtered)
print(f"Heart rate: {heart_rate(filtered, peaks):.0f} bpm")

# Visualize
fig = plot_12lead(record)
fig.savefig("ecg_report.png", dpi=150)
```

```{raw} html
<!-- ── Supported Formats ──────────────────────────────── -->
<h2 class="landing-section-title">Supported Formats</h2>
<div class="landing-formats">
  <div class="landing-format"><strong>HL7 aECG</strong><span>Health Level 7 annotated ECG</span></div>
  <div class="landing-format"><strong>Philips Sierra XML</strong><span>Philips Sierra ECG format</span></div>
  <div class="landing-format"><strong>GE MUSE XML</strong><span>GE MUSE ECG management</span></div>
  <div class="landing-format"><strong>ISHNE Holter</strong><span>Holter &amp; Noninvasive ECG</span></div>
  <div class="landing-format"><strong>Mortara EL250</strong><span>Mortara ELI 250 device</span></div>
  <div class="landing-format"><strong>EDF/EDF+</strong><span>European Data Format</span></div>
  <div class="landing-format"><strong>SCP-ECG</strong><span>Standard Comms Protocol</span></div>
  <div class="landing-format"><strong>DICOM Waveform</strong><span>Medical imaging standard</span></div>
  <div class="landing-format"><strong>WFDB</strong><span>PhysioNet WaveForm DB</span></div>
  <div class="landing-format"><strong>MFER</strong><span>Medical waveform encoding</span></div>
  <div class="landing-format"><strong>Mindray R12</strong><span>BeneHeart R12 device</span></div>
  <div class="landing-format"><strong>GE MAC 2000</strong><span>Resting ECG system</span></div>
</div>

<!-- ── Installation ───────────────────────────────────── -->
<h2 class="landing-section-title">Installation</h2>
```

```bash
# Core (parsing only)
pip install .

# With signal processing (scipy)
pip install ".[processing]"

# With static plots (matplotlib)
pip install ".[plotting]"

# With interactive plots (plotly)
pip install ".[plotting-interactive]"

# Everything
pip install ".[all]"
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Guides

guides/getting-started
guides/formats
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

api/parsing
api/processing
api/plotting
api/reference
api/models
api/exceptions
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Project

releases
```
