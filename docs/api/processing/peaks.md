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

{func}`~ecgdatakit.processing.detect_r_peaks` supports two algorithms, selected via the `method` parameter.

##### Pan-Tompkins (`"pan_tompkins"`, default)

Classic real-time QRS detection algorithm. Applies a cascade of signal conditioning stages followed by adaptive thresholding:

1. **Bandpass filter (5–15 Hz)** — 2nd-order Butterworth isolates the QRS frequency range while suppressing P/T waves and high-frequency noise.
2. **Five-point derivative** — kernel `[-1, -2, 0, 2, 1]` scaled by `fs/8` emphasises the steep QRS slopes.
3. **Squaring** — amplifies large differences and ensures all values are positive.
4. **Moving-window integration (150 ms)** — smooths the squared derivative into a single energy pulse per QRS.
5. **Adaptive dual-threshold** — maintains running estimates of signal and noise peak levels. A primary threshold `T1 = Npk + 0.25 × (Spk − Npk)` detects most beats; a lower secondary threshold `T2 = 0.5 × T1` triggers a **searchback** when the RR interval exceeds 166 % of the running average, recovering missed beats.
6. **Refinement** — each candidate is relocated to the local maximum of the original signal within ±75 ms.

```{tip}
Best for clean or moderately noisy recordings. The adaptive thresholds and searchback mechanism handle gradual amplitude changes well.
```

##### Shannon energy envelope (`"shannon_energy"`)

Entropy-based detector that is more robust to noise and amplitude variations:

1. **Narrow bandpass (35–43 Hz)** — two cascaded 1st-order Butterworth filters extract the high-frequency QRS energy while rejecting baseline wander and T waves.
2. **Derivative power** — first difference, then squaring.
3. **RANSAC adaptive threshold** — the signal is divided into 5-second windows; per-window standard deviations and maxima are computed, and the **median** is taken as a robust global threshold — effectively ignoring outlier windows.
4. **Shannon entropy transform** — for normalised power values $x \in [0, 1]$: $E = -x^2 \ln(x^2)$. This nonlinear transform attenuates large and small values alike, enhancing medium-amplitude peaks and compressing outliers.
5. **Envelope smoothing** — 125 ms moving average followed by Gaussian smoothing ($\sigma = f_s / 8$).
6. **Zero-crossing peak detection** — peaks of the smoothed envelope are located at positive-to-negative zero-crossings of its first derivative.
7. **Refinement** — same ±75 ms relocation in the original signal.

```{tip}
Preferred for noisy signals, low-amplitude QRS complexes, or recordings with large amplitude variations. The Shannon entropy transform naturally normalises energy, making it less sensitive to signal amplitude.
```

##### Comparison

| | Pan-Tompkins | Shannon Energy |
|---|---|---|
| **Bandpass** | 5–15 Hz (broadband QRS) | 35–43 Hz (high-frequency QRS) |
| **Thresholding** | Dual adaptive + searchback | RANSAC on 5 s windows |
| **Core feature** | Moving-window integrated energy | Shannon entropy envelope |
| **Strength** | Fast, well-tested on clean ECG | Robust to noise and amplitude variation |
| **Best for** | Standard clinical recordings | Ambulatory / noisy / low-amplitude signals |

```python
from ecgdatakit.processing import detect_r_peaks, heart_rate, rr_intervals

# Pan-Tompkins (default)
peaks = detect_r_peaks(lead)

# Shannon energy — better for noisy signals
peaks_se = detect_r_peaks(lead, method="shannon_energy")

# Downstream metrics
hr = heart_rate(lead, peaks)          # e.g. 72.5 bpm
rr = rr_intervals(lead, peaks)       # array of RR in ms
```

---

```{eval-rst}
.. autofunction:: heart_rate
.. autofunction:: rr_intervals
.. autofunction:: instantaneous_heart_rate
```
