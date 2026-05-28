# Filters

All filters use SOS (second-order sections) + zero-phase `sosfiltfilt` to preserve ECG morphology.

| | |
|---|---|
| {func}`~ecgdatakit.processing.lowpass` | Apply a Butterworth low-pass filter |
| {func}`~ecgdatakit.processing.highpass` | Apply a Butterworth high-pass filter |
| {func}`~ecgdatakit.processing.bandpass` | Apply a Butterworth band-pass filter |
| {func}`~ecgdatakit.processing.notch` | Apply an IIR notch (band-stop) filter |
| {func}`~ecgdatakit.processing.remove_baseline` | Remove baseline wander using a high-pass filter |
| {func}`~ecgdatakit.processing.diagnostic_filter` | Apply AHA diagnostic-grade filtering: 0.05–150 Hz bandpass + notch |
| {func}`~ecgdatakit.processing.monitoring_filter` | Apply monitoring-grade filtering: 0.67–40 Hz bandpass + notch |

```{eval-rst}
.. currentmodule:: ecgdatakit.processing

.. autofunction:: lowpass
.. autofunction:: highpass
.. autofunction:: bandpass
.. autofunction:: notch
.. autofunction:: remove_baseline
.. autofunction:: diagnostic_filter
.. autofunction:: monitoring_filter
```
