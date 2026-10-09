# Resampling

| | |
|---|---|
| {func}`~ecgdatakit.processing.resample` | Resample a lead to a different sample rate |

Polyphase resampling with linear edge extension, so a DC level or slow drift does not ring at the ends. The output has `ceil(n * target_rate / fs)` samples and the returned lead's `sampling_rate` is updated. `target_rate` must be a whole number of Hz (`250` or `250.0`). Input must be a single finite lead (1-D); NaN or Inf samples raise `ValueError`.

```{eval-rst}
.. currentmodule:: ecgdatakit.processing

.. autofunction:: resample
```
