# R-Peak Detection

| | |
|---|---|
| {func}`~ecgdatakit.processing.detect_r_peaks` | Detect R-peak locations in an ECG lead |
| {func}`~ecgdatakit.processing.heart_rate` | Compute average heart rate in beats per minute |
| {func}`~ecgdatakit.processing.rr_intervals` | Compute RR intervals in milliseconds |
| {func}`~ecgdatakit.processing.instantaneous_heart_rate` | Compute instantaneous heart rate at each beat |

```{eval-rst}
.. currentmodule:: ecgdatakit.processing

.. autofunction:: detect_r_peaks
```

#### Detection methods

{func}`~ecgdatakit.processing.detect_r_peaks` supports two algorithms, selected via the `method` parameter. Both work at any sample rate (all time constants are in seconds), accept either QRS polarity, and return sorted, unique sample indices. The signal must be 1-D, finite and at least 1 s long; otherwise a `ValueError` is raised.

Exactly constant runs of 2 s or more (zero padding at the end of a short recording, a disconnected or saturated lead) are skipped, and the stretches between them are processed separately, so the step into the padding is not taken as a beat. ECG stretches shorter than 1 s between such runs are not analysed. A constant signal returns no peaks.

Peak indices always refer to the full signal. The beats on either side of a skipped run are therefore consecutive in {func}`~ecgdatakit.processing.rr_intervals`, and their RR interval spans the whole gap: 10 s of ECG, 5 s flat, then 10 s of ECG gives one RR interval longer than 5 s (for example 5.76 s when the beats on either side lie 0.76 s in total from the gap edges). This interval also lowers {func}`~ecgdatakit.processing.heart_rate`. Remove it before HRV analysis.

##### Shannon energy envelope (`"shannon_energy"`)

1. **Narrow bandpass (35–43 Hz)**: two cascaded 1st-order Butterworth filters, applied forward and backward, extract the high-frequency QRS energy while rejecting baseline wander and T waves.
2. **Derivative power**: first difference, then squaring.
3. **Windowed adaptive threshold**: per 5-second window, a threshold (half the standard deviation) and a scale (the maximum) are computed and smoothed with a 3-window median, so a single artefact window is ignored while slow amplitude changes are followed.
4. **Shannon energy transform**: for normalised power values $x \in [0, 1]$: $E = -x^2 \ln(x^2)$.
5. **Envelope smoothing**: 125 ms moving average followed by Gaussian smoothing ($\sigma = f_s / 8$).
6. **Peak picking**: maxima of the smoothed envelope (positive-to-negative zero-crossings of its derivative) above 2 % of the local envelope level.
7. **Refinement**: see below.

##### Pan-Tompkins (`"pan_tompkins"`, default)

1. **Bandpass filter (5–15 Hz)**: 2nd-order Butterworth, zero-phase.
2. **Derivative**: central difference scaled by $f_s$.
3. **Squaring**.
4. **Moving-window integration (150 ms)**.
5. **Learning phase**: the signal and noise levels start from the median of the maxima and means of 2-second windows, so one artefact cannot set the threshold. No beat is accepted below 1 % of the median window maximum (or below an absolute floor far above rounding noise). Such small peaks still update the noise level, so the thresholds keep following the signal. A signal whose bandpassed energy is negligible returns no peaks.
6. **Adaptive dual threshold**: `T1 = Npk + 0.25 × (Spk − Npk)` on integrated-signal peaks at least 200 ms apart. Peak heights are capped at 4 × the initial signal level when updating it, so a burst of artefacts cannot lock the detector. A peak within 360 ms of the previous beat whose maximum slope is under half that beat's slope is treated as a T wave.
7. **Searchback**: when no beat is found for 1.66 × the median of the last 8 RR intervals, the largest skipped peak above `T2 = 0.5 × T1` is taken as a beat. Skipped peaks that fail the T-wave test are never taken. If none qualifies, the signal level is halved (never below the floor) and the search is repeated once, so the detector recovers after a drop in amplitude.
8. **Refinement**: see below.

##### Refinement

Each candidate is moved to the dominant QRS deflection within ±75 ms in a 0.5–40 Hz filtered copy of the signal. The polarity is decided once per lead (median positive versus median negative excursion), so on leads with a mainly negative QRS (aVR, often V1) the peak is placed on the negative deflection. Candidates closer than 200 ms are merged.

##### Benchmark

Measured on 18 MIT-BIH Arrhythmia Database records (100, 101, 103, 105, 106, 108, 109, 114, 119, 200, 203, 207, 208, 210, 213, 222, 228, 232; 30 min each, first channel, 41 595 annotated beats). A detection counts as correct within 150 ms of an annotation. Signals were resampled from 360 Hz for the other rates.

| Sample rate | Shannon energy Se / PPV | Pan-Tompkins Se / PPV |
|---|---|---|
| 250 Hz | 0.976 / 0.977 | 0.995 / 0.982 |
| 360 Hz | 0.976 / 0.977 | 0.995 / 0.981 |
| 500 Hz | 0.976 / 0.977 | 0.995 / 0.981 |
| 1000 Hz | 0.976 / 0.977 | 0.995 / 0.981 |

Results are identical for inverted signals. For comparison, at 360 Hz `wfdb.processing.xqrs_detect` scores 0.985 / 0.982 and NeuroKit2 `ecg_peaks` 0.973 / 0.991 on the same records.

##### Limitations

- Flat or lead-off stretches are recognised only when they are exactly constant for 2 s or more. A noisy flat line is treated as a low-amplitude signal and may produce false beats.
- Square-wave or step artefacts (lead-off pulses, electrode pops, gain switching) are not recognised. Pan-Tompkins keeps finding the beats around them but reports their edges as beats. With Shannon energy, an artefact that fills a large part of a 5-second window can also hide the real beats in that window and its neighbours.
- A T wave that is both taller and nearly as sharp as the QRS (for example taller than R and under about 70 ms wide) can pass the T-wave slope test and be counted as a beat.
- No ectopic or noise classification is done: every detected complex is returned.

```python
from ecgdatakit.processing import detect_r_peaks, heart_rate, rr_intervals

# Pan-Tompkins (default)
peaks = detect_r_peaks(lead)

# Shannon energy
peaks_se = detect_r_peaks(lead, method="shannon_energy")

# Downstream metrics
hr = heart_rate(lead, peaks)          # e.g. 72.5 bpm (nan if < 2 peaks)
rr = rr_intervals(lead, peaks)       # array of RR in ms
```

---

```{eval-rst}
.. autofunction:: heart_rate
.. autofunction:: rr_intervals
.. autofunction:: instantaneous_heart_rate
```
