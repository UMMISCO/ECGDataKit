# Signal Quality

| | |
|---|---|
| {func}`~ecgdatakit.processing.signal_quality_index` | Compute a composite signal quality index (SQI) in [0, 1] |
| {func}`~ecgdatakit.processing.classify_quality` | Classify signal quality as a human-readable category |
| {func}`~ecgdatakit.processing.snr_estimate` | Estimate signal-to-noise ratio in dB |

The SQI is a heuristic screening score, not a validated clinical measure. It is the mean of four sub-scores, each in [0, 1]. Band components are obtained with zero-phase Butterworth filters, so the scores do not depend on the sample rate. The signal must be 1-D, finite and at least 1 s long.

| Sub-score | Definition | Scoring |
|---|---|---|
| Kurtosis | Excess kurtosis of the signal (after Li, Clark and Clifford 2008) | 1 for 3–20; `k / 3` below 3; 0 when negative; linear decrease to 0 at 50 above 20 |
| Power ratio | Variance of the 1–40 Hz band / variance of the mean-removed signal | `ratio / 0.7`, clipped to [0, 1] |
| Peak regularity | Coefficient of variation (CV) of RR intervals from {func}`~ecgdatakit.processing.detect_r_peaks` | `1 − 2 × CV`, clipped; 0.3 when fewer than 3 peaks |
| Baseline stability | Variance below 1 Hz / variance of the 1–40 Hz band | `1 − ratio`, clipped |

{func}`~ecgdatakit.processing.classify_quality` maps the SQI to `"excellent"` (> 0.8), `"acceptable"` (0.5–0.8) or `"unacceptable"` (< 0.5).

```{note}
Irregular rhythms (atrial fibrillation, frequent ectopy) lower the peak-regularity score even when the recording is clean.
```

**SNR estimate.** {func}`~ecgdatakit.processing.snr_estimate` takes the noise floor as the mean power spectral density above 100 Hz (Welch, 1 s segments), assumes it is white, and extrapolates it over the 1–40 Hz ECG band. The result is the 1–40 Hz power divided by that noise power, in dB. Limits:

- It returns `nan` at 200 Hz or below, where there is no band above 100 Hz (the half-weight Nyquist bin is excluded from the noise band).
- It returns `inf` when no noise is measured.
- It measures only broadband noise reaching above 100 Hz (EMG, electronic noise). Baseline wander, powerline interference and in-band motion artefacts are not counted.
- Recordings low-pass filtered below 100 Hz by the device read as noise-free.

```{eval-rst}
.. currentmodule:: ecgdatakit.processing

.. autofunction:: signal_quality_index
.. autofunction:: classify_quality
.. autofunction:: snr_estimate
```
