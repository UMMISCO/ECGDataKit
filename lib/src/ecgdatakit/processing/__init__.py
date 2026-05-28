"""ECG signal processing utilities.

Requires: ``pip install ecgdatakit[processing]``

Modules
-------
filters
    Butterworth low/high/band-pass, notch, baseline removal, presets.
resample
    Polyphase resampling.
normalize
    Min-max, z-score, and amplitude normalization.
peaks
    R-peak detection (Pan-Tompkins, Shannon energy), heart rate, RR intervals.
hrv
    Time-domain and frequency-domain heart rate variability.
transforms
    FFT, PSD (Welch), beat segmentation, ensemble averaging.
quality
    Signal quality index, SNR estimation.
leads
    Lead derivation (III, aVR/aVL/aVF, full 12-lead assembly).
clean
    ECG cleaning (built-in, BioSPPy, NeuroKit2, combined, DeepFADE).
"""

from ecgdatakit.processing.clean import clean_ecg
from ecgdatakit.processing.filters import (
    bandpass,
    diagnostic_filter,
    highpass,
    lowpass,
    monitoring_filter,
    notch,
    remove_baseline,
)
from ecgdatakit.processing.hrv import frequency_domain, poincare, time_domain
from ecgdatakit.processing.leads import (
    derive_augmented,
    derive_lead_iii,
    derive_standard_12,
    find_lead,
)
from ecgdatakit.processing.normalize import (
    normalize_amplitude,
    normalize_minmax,
    normalize_zscore,
)
from ecgdatakit.processing.peaks import (
    detect_r_peaks,
    heart_rate,
    instantaneous_heart_rate,
    rr_intervals,
)
from ecgdatakit.processing.quality import (
    classify_quality,
    signal_quality_index,
    snr_estimate,
)
from ecgdatakit.processing.resample import resample
from ecgdatakit.processing.transforms import (
    average_beat,
    fft,
    power_spectrum,
    segment_beats,
)

__all__ = [
    "clean_ecg",
    "lowpass",
    "highpass",
    "bandpass",
    "notch",
    "remove_baseline",
    "diagnostic_filter",
    "monitoring_filter",
    "resample",
    "normalize_minmax",
    "normalize_zscore",
    "normalize_amplitude",
    "detect_r_peaks",
    "heart_rate",
    "rr_intervals",
    "instantaneous_heart_rate",
    "time_domain",
    "frequency_domain",
    "poincare",
    "power_spectrum",
    "fft",
    "segment_beats",
    "average_beat",
    "signal_quality_index",
    "classify_quality",
    "snr_estimate",
    "derive_lead_iii",
    "derive_augmented",
    "derive_standard_12",
    "find_lead",
]
