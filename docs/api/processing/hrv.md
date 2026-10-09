# Heart Rate Variability

| | |
|---|---|
| {func}`~ecgdatakit.processing.time_domain` | Compute time-domain HRV metrics from RR intervals |
| {func}`~ecgdatakit.processing.frequency_domain` | Compute frequency-domain HRV metrics from RR intervals |
| {func}`~ecgdatakit.processing.poincare` | Compute Poincaré plot descriptors (SD1, SD2) |

Definitions follow the Task Force of the ESC and NASPE (1996), *Heart rate variability: standards of measurement, physiological interpretation and clinical use*, Circulation 93:1043–1065.

- Input is a 1-D array of RR intervals in **milliseconds**. NaN, infinite or non-positive values raise `ValueError`.
- The intervals are used as given. Ectopic beats and detection errors are not removed, so pass cleaned NN intervals when the result must follow the standard.
- A metric that is undefined for the number of intervals is returned as `nan`: SDNN and RMSSD need 2 intervals, SDSD needs 3, frequency-domain metrics need 4, SD1/SD2 need 3.

| Metric | Unit | Definition |
|---|---|---|
| `sdnn` | ms | Sample standard deviation of the intervals (ddof = 1) |
| `rmssd` | ms | Root mean square of successive differences |
| `sdsd` | ms | Sample standard deviation of successive differences |
| `nn50_count`, `pnn50` | count, % | Successive differences **strictly greater** than 50 ms, as a percentage of all differences |
| `nn20_count`, `pnn20` | count, % | Same with 20 ms |
| `hr_mean`, `hr_std` | bpm | Mean and sample standard deviation of `60000 / RR` |
| `vlf_power`, `lf_power`, `hf_power`, `total_power` | ms² | Welch PSD of the RR series, cubic-interpolated at `interp_fs` (default 4 Hz, must be finite and above 0.8 Hz), integrated over 0–0.04, 0.04–0.15, 0.15–0.40 and 0–0.40 Hz |
| `lf_hf_ratio` | – | `lf_power / hf_power` (`nan` when HF is 0) |
| `sd1`, `sd2` | ms | `std(RR[n+1] − RR[n]) / √2` and `std(RR[n+1] + RR[n]) / √2` |

```{warning}
Frequency-domain metrics need enough data: the Task Force recommends at least 1 min for HF, 2 min for LF and 5 min for short-term analysis. {func}`~ecgdatakit.processing.frequency_domain` issues a `UserWarning` below 2 min (LF unreliable) and below 1 min (LF and HF unreliable). The RR intervals of a standard 10-second ECG are not enough for spectral HRV. Do not interpret VLF from recordings shorter than 5 min.
```

```{eval-rst}
.. currentmodule:: ecgdatakit.processing

.. autofunction:: time_domain
.. autofunction:: frequency_domain
.. autofunction:: poincare
```
