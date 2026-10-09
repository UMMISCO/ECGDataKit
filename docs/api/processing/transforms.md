# Transforms & Segmentation

| | |
|---|---|
| {func}`~ecgdatakit.processing.power_spectrum` | Compute the power spectral density of an ECG lead |
| {func}`~ecgdatakit.processing.fft` | Compute the single-sided FFT magnitude spectrum |
| {func}`~ecgdatakit.processing.segment_beats` | Segment individual heartbeats around R-peaks |
| {func}`~ecgdatakit.processing.average_beat` | Compute the ensemble-averaged heartbeat (template) |

- `fft` returns a single-sided amplitude spectrum: a sinusoid of amplitude A reads A, and the DC (and even-length Nyquist) bins are not doubled.
- `power_spectrum` supports `method="welch"` (default) and `method="periodogram"`; any other value raises `ValueError`.
- `fft` and `power_spectrum` refuse empty input and NaN or Inf samples.
- `segment_beats` and `average_beat` detect R-peaks with the default method of {func}`~ecgdatakit.processing.detect_r_peaks` when `peaks` is not given. Peaks must be integer indices (whole-number floats are accepted).
- `average_beat` raises `ValueError` when no complete beat is available, instead of returning a flat template.

```{eval-rst}
.. currentmodule:: ecgdatakit.processing

.. autofunction:: power_spectrum
.. autofunction:: fft
.. autofunction:: segment_beats
.. autofunction:: average_beat
```
