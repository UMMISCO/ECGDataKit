---
title: "Function Reference"
weight: 40
---

Detailed reference for every public function in ECGDataKit.

## Processing

### ECG Cleaning

- [`clean_ecg`](/api/ref/clean_ecg/)

### Filters

- [`lowpass`](/api/ref/lowpass/)
- [`highpass`](/api/ref/highpass/)
- [`bandpass`](/api/ref/bandpass/)
- [`notch`](/api/ref/notch/)
- [`remove_baseline`](/api/ref/remove_baseline/)
- [`diagnostic_filter`](/api/ref/diagnostic_filter/)
- [`monitoring_filter`](/api/ref/monitoring_filter/)

### Resampling

- [`resample`](/api/ref/resample/)

### Normalization

- [`normalize_minmax`](/api/ref/normalize_minmax/)
- [`normalize_zscore`](/api/ref/normalize_zscore/)
- [`normalize_amplitude`](/api/ref/normalize_amplitude/)

### R-Peak Detection

- [`detect_r_peaks`](/api/ref/detect_r_peaks/)
- [`heart_rate`](/api/ref/heart_rate/)
- [`rr_intervals`](/api/ref/rr_intervals/)
- [`instantaneous_heart_rate`](/api/ref/instantaneous_heart_rate/)

### Heart Rate Variability

- [`time_domain`](/api/ref/time_domain/)
- [`frequency_domain`](/api/ref/frequency_domain/)
- [`poincare`](/api/ref/poincare/)

### Transforms & Segmentation

- [`power_spectrum`](/api/ref/power_spectrum/)
- [`fft`](/api/ref/fft/)
- [`segment_beats`](/api/ref/segment_beats/)
- [`average_beat`](/api/ref/average_beat/)

### Signal Quality

- [`signal_quality_index`](/api/ref/signal_quality_index/)
- [`classify_quality`](/api/ref/classify_quality/)
- [`snr_estimate`](/api/ref/snr_estimate/)

### Lead Derivation

- [`derive_lead_iii`](/api/ref/derive_lead_iii/)
- [`derive_augmented`](/api/ref/derive_augmented/)
- [`derive_standard_12`](/api/ref/derive_standard_12/)
- [`find_lead`](/api/ref/find_lead/)

## Plotting

### Static Plots (matplotlib)

- [`plot_lead`](/api/ref/plot_lead/)
- [`plot_leads`](/api/ref/plot_leads/)
- [`plot_12lead`](/api/ref/plot_12lead/)
- [`plot_peaks`](/api/ref/plot_peaks/)
- [`plot_beats`](/api/ref/plot_beats/)
- [`plot_average_beat`](/api/ref/plot_average_beat/)
- [`plot_spectrum`](/api/ref/plot_spectrum/)
- [`plot_spectrogram`](/api/ref/plot_spectrogram/)
- [`plot_rr_tachogram`](/api/ref/plot_rr_tachogram/)
- [`plot_poincare`](/api/ref/plot_poincare/)
- [`plot_hrv_summary`](/api/ref/plot_hrv_summary/)
- [`plot_quality`](/api/ref/plot_quality/)
- [`plot_report`](/api/ref/plot_report/)

### Interactive Plots (plotly)

- [`iplot_lead`](/api/ref/iplot_lead/)
- [`iplot_leads`](/api/ref/iplot_leads/)
- [`iplot_12lead`](/api/ref/iplot_12lead/)
- [`iplot_peaks`](/api/ref/iplot_peaks/)
- [`iplot_spectrum`](/api/ref/iplot_spectrum/)
- [`iplot_rr_tachogram`](/api/ref/iplot_rr_tachogram/)
- [`iplot_poincare`](/api/ref/iplot_poincare/)
- [`iplot_report`](/api/ref/iplot_report/)
