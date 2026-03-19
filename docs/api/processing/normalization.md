# Normalization

Pure numpy — no scipy required.

All functions accept a single lead (`Lead` or numpy array) **or** a `list[Lead]` for per-lead normalization across multiple leads.

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
