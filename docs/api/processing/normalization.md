# Normalization

Pure numpy, no scipy required.

All functions accept a `Lead`, a `list[Lead]`, an `ECGRecord` (leads and median beats), a `list[ECGRecord]`, or a numpy array of shape `(n_samples,)`, `(n_leads, n_samples)` or `(n_ecgs, n_leads, n_samples)`. Each lead (each row along the last axis) is normalized independently, and numpy results are float64.

- `normalize_minmax` and `normalize_zscore` return dimensionless leads: `units=""`, `resolution=1.0`, `resolution_unit=""`, `offset=0.0`, `is_raw=False`.
- `normalize_amplitude` is a pure gain and keeps units and scaling metadata. For a lead in a known voltage unit the peak becomes `target_mv` millivolts in that unit (1000 for a `"uV"` lead); raw leads are scaled through their `resolution`. Without a known unit, and for numpy input, the peak becomes `target_mv` in the signal's own units.
- NaN samples are ignored when computing the statistics and stay NaN. A constant signal normalizes to zeros. A peak-to-peak range of at most `1e-12` times the largest magnitude counts as constant, so rounding noise (e.g. `[0.1 + 0.2, 0.3]`) is not scaled up to ±1. A flat line around zero, such as a high-passed constant, has no reference scale and is normalized like any other signal.

| | |
|---|---|
| {func}`~ecgdatakit.processing.normalize_minmax` | Scale signal to the [−1, 1] range |
| {func}`~ecgdatakit.processing.normalize_zscore` | Normalize to zero mean and unit variance (z-score) |
| {func}`~ecgdatakit.processing.normalize_amplitude` | Scale peak amplitude to a target value |

```{eval-rst}
.. currentmodule:: ecgdatakit.processing

.. autofunction:: normalize_minmax
.. autofunction:: normalize_zscore
.. autofunction:: normalize_amplitude
```
