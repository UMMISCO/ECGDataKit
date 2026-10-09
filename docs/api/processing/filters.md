# Filters

All filters use SOS (second-order sections) + zero-phase `sosfiltfilt` to preserve ECG morphology.

- Input must be a single finite lead (1-D). NaN or Inf samples raise `ValueError`, since one NaN would spread over the whole filtered output. Signals shorter than the filter's edge padding (27 samples for a 4th-order band-pass) also raise `ValueError`.
- Raw leads are filtered in ADC counts. Their `offset` is folded into the samples, so `to_physical()` after filtering gives the same result as filtering the physical signal.
- `diagnostic_filter` and `monitoring_filter` lower their upper cutoff to `0.45 * fs` when the sampling rate is too low (e.g. 81 Hz instead of 150 Hz at 180 Hz), and skip a notch that is not below Nyquist. Both cases emit a warning that points at your code. `clean_ecg(method="default")` applies the same rule to its 40 Hz cutoff (27 Hz at 60 Hz). Pass `notch_freq=None` to disable the notch, or `notch_freq=60` for 60 Hz mains.

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
