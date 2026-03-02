"""Tests for numpy array input support across processing and plotting functions.

Every function that accepts LeadLike should:
1. Work with a Lead object (backward compat — covered by existing tests)
2. Work with a numpy array + fs keyword
3. Raise TypeError when numpy array is passed without fs
"""

import numpy as np
import pytest
from ecgdatakit.models import Lead

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_signal(fs=500, duration=2.0, freq=10.0):
    """Return a pure sine wave as a numpy float64 array."""
    t = np.arange(0, duration, 1.0 / fs)
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


# ---------------------------------------------------------------------------
# ensure_lead (processing)
# ---------------------------------------------------------------------------

class TestEnsureLeadProcessing:
    def test_lead_passthrough(self):
        from ecgdatakit.processing._core import ensure_lead
        lead = Lead(label="II", samples=_sine_signal(), sample_rate=500)
        assert ensure_lead(lead) is lead

    def test_numpy_with_fs(self):
        from ecgdatakit.processing._core import ensure_lead
        arr = _sine_signal()
        result = ensure_lead(arr, fs=500)
        assert isinstance(result, Lead)
        assert result.sample_rate == 500
        assert np.array_equal(result.samples, arr)

    def test_numpy_without_fs_raises(self):
        from ecgdatakit.processing._core import ensure_lead
        arr = _sine_signal()
        with pytest.raises(TypeError, match="fs"):
            ensure_lead(arr)

    def test_list_input(self):
        from ecgdatakit.processing._core import ensure_lead
        result = ensure_lead([1.0, 2.0, 3.0], fs=100)
        assert isinstance(result, Lead)
        assert result.samples.dtype == np.float64

    def test_custom_label(self):
        from ecgdatakit.processing._core import ensure_lead
        result = ensure_lead(_sine_signal(), fs=500, label="V1")
        assert result.label == "V1"


# ---------------------------------------------------------------------------
# ensure_lead (plotting)
# ---------------------------------------------------------------------------

class TestEnsureLeadPlotting:
    def test_lead_passthrough(self):
        from ecgdatakit.plotting._core import ensure_lead
        lead = Lead(label="II", samples=_sine_signal(), sample_rate=500)
        assert ensure_lead(lead) is lead

    def test_numpy_with_fs(self):
        from ecgdatakit.plotting._core import ensure_lead
        arr = _sine_signal()
        result = ensure_lead(arr, fs=500)
        assert isinstance(result, Lead)
        assert result.sample_rate == 500

    def test_numpy_without_fs_raises(self):
        from ecgdatakit.plotting._core import ensure_lead
        with pytest.raises(TypeError, match="fs"):
            ensure_lead(_sine_signal())


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFiltersNumpyInput:
    def test_lowpass(self):
        from ecgdatakit.processing.filters import lowpass
        result = lowpass(_sine_signal(), cutoff=50, fs=500)
        assert isinstance(result, Lead)
        assert result.sample_rate == 500

    def test_lowpass_no_fs_raises(self):
        from ecgdatakit.processing.filters import lowpass
        with pytest.raises(TypeError, match="fs"):
            lowpass(_sine_signal(), cutoff=50)

    def test_highpass(self):
        from ecgdatakit.processing.filters import highpass
        result = highpass(_sine_signal(duration=5.0), cutoff=1.0, fs=500)
        assert isinstance(result, Lead)

    def test_bandpass(self):
        from ecgdatakit.processing.filters import bandpass
        result = bandpass(_sine_signal(duration=5.0), low=1.0, high=50.0, fs=500)
        assert isinstance(result, Lead)

    def test_notch(self):
        from ecgdatakit.processing.filters import notch
        result = notch(_sine_signal(), fs=500)
        assert isinstance(result, Lead)

    def test_remove_baseline(self):
        from ecgdatakit.processing.filters import remove_baseline
        result = remove_baseline(_sine_signal(duration=5.0), fs=500)
        assert isinstance(result, Lead)

    def test_diagnostic_filter(self):
        from ecgdatakit.processing.filters import diagnostic_filter
        result = diagnostic_filter(_sine_signal(), fs=500)
        assert isinstance(result, Lead)

    def test_monitoring_filter(self):
        from ecgdatakit.processing.filters import monitoring_filter
        result = monitoring_filter(_sine_signal(), fs=500)
        assert isinstance(result, Lead)


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------

class TestResampleNumpyInput:
    def test_resample(self):
        from ecgdatakit.processing.resample import resample
        result = resample(_sine_signal(), target_rate=250, fs=500)
        assert isinstance(result, Lead)
        assert result.sample_rate == 250

    def test_resample_no_fs_raises(self):
        from ecgdatakit.processing.resample import resample
        with pytest.raises(TypeError, match="fs"):
            resample(_sine_signal(), target_rate=250)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class TestNormalizeNumpyInput:
    def test_minmax(self):
        from ecgdatakit.processing.normalize import normalize_minmax
        result = normalize_minmax(_sine_signal(), fs=500)
        assert isinstance(result, Lead)
        assert result.samples.max() <= 1.0
        assert result.samples.min() >= -1.0

    def test_zscore(self):
        from ecgdatakit.processing.normalize import normalize_zscore
        result = normalize_zscore(_sine_signal(), fs=500)
        assert isinstance(result, Lead)
        assert abs(result.samples.mean()) < 1e-10

    def test_amplitude(self):
        from ecgdatakit.processing.normalize import normalize_amplitude
        result = normalize_amplitude(_sine_signal(), target_mv=2.0, fs=500)
        assert isinstance(result, Lead)
        assert abs(np.abs(result.samples).max() - 2.0) < 1e-10

    def test_minmax_no_fs_raises(self):
        from ecgdatakit.processing.normalize import normalize_minmax
        with pytest.raises(TypeError, match="fs"):
            normalize_minmax(_sine_signal())


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------

class TestPeaksNumpyInput:
    def _ecg_like_signal(self, fs=500):
        """Generate a signal with sharp peaks for R-peak detection."""
        t = np.arange(0, 5.0, 1.0 / fs)
        # Create periodic sharp peaks
        signal = np.zeros_like(t)
        period = int(0.8 * fs)  # 75 bpm
        for i in range(0, len(t), period):
            if i < len(signal):
                signal[i] = 1.0
        # Smooth a bit
        from scipy.ndimage import gaussian_filter1d
        signal = gaussian_filter1d(signal, sigma=3)
        return signal.astype(np.float64)

    def test_detect_r_peaks(self):
        from ecgdatakit.processing.peaks import detect_r_peaks
        signal = self._ecg_like_signal()
        peaks = detect_r_peaks(signal, fs=500)
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) > 0

    def test_heart_rate(self):
        from ecgdatakit.processing.peaks import heart_rate
        signal = self._ecg_like_signal()
        hr = heart_rate(signal, fs=500)
        assert isinstance(hr, float)

    def test_rr_intervals(self):
        from ecgdatakit.processing.peaks import rr_intervals
        signal = self._ecg_like_signal()
        rr = rr_intervals(signal, fs=500)
        assert isinstance(rr, np.ndarray)

    def test_detect_r_peaks_no_fs_raises(self):
        from ecgdatakit.processing.peaks import detect_r_peaks
        with pytest.raises(TypeError, match="fs"):
            detect_r_peaks(self._ecg_like_signal())


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestTransformsNumpyInput:
    def test_power_spectrum(self):
        from ecgdatakit.processing.transforms import power_spectrum
        freqs, power = power_spectrum(_sine_signal(), fs=500)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(power, np.ndarray)

    def test_fft(self):
        from ecgdatakit.processing.transforms import fft
        freqs, mags = fft(_sine_signal(), fs=500)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(mags, np.ndarray)

    def test_power_spectrum_no_fs_raises(self):
        from ecgdatakit.processing.transforms import power_spectrum
        with pytest.raises(TypeError, match="fs"):
            power_spectrum(_sine_signal())


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

class TestQualityNumpyInput:
    def test_signal_quality_index(self):
        from ecgdatakit.processing.quality import signal_quality_index
        sqi = signal_quality_index(_sine_signal(), fs=500)
        assert isinstance(sqi, float)
        assert 0.0 <= sqi <= 1.0

    def test_classify_quality(self):
        from ecgdatakit.processing.quality import classify_quality
        result = classify_quality(_sine_signal(), fs=500)
        assert result in ("excellent", "acceptable", "unacceptable")

    def test_snr_estimate(self):
        from ecgdatakit.processing.quality import snr_estimate
        snr = snr_estimate(_sine_signal(), fs=500)
        assert isinstance(snr, float)

    def test_signal_quality_no_fs_raises(self):
        from ecgdatakit.processing.quality import signal_quality_index
        with pytest.raises(TypeError, match="fs"):
            signal_quality_index(_sine_signal())


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

class TestCleanNumpyInput:
    def test_clean_default(self):
        from ecgdatakit.processing.clean import clean_ecg
        result = clean_ecg(_sine_signal(), method="default", fs=500)
        assert isinstance(result, Lead)
        assert result.sample_rate == 500

    def test_clean_no_fs_raises(self):
        from ecgdatakit.processing.clean import clean_ecg
        with pytest.raises(TypeError, match="fs"):
            clean_ecg(_sine_signal())


# ---------------------------------------------------------------------------
# Static Plotting
# ---------------------------------------------------------------------------

class TestStaticPlottingNumpyInput:
    def test_plot_lead(self):
        from ecgdatakit.plotting import plot_lead
        fig = plot_lead(_sine_signal(), fs=500)
        assert fig is not None

    def test_plot_lead_no_fs_raises(self):
        from ecgdatakit.plotting import plot_lead
        with pytest.raises(TypeError, match="fs"):
            plot_lead(_sine_signal())

    def test_plot_spectrum(self):
        from ecgdatakit.plotting import plot_spectrum
        fig = plot_spectrum(_sine_signal(), fs=500)
        assert fig is not None

    def test_plot_spectrogram(self):
        from ecgdatakit.plotting import plot_spectrogram
        fig = plot_spectrogram(_sine_signal(), fs=500)
        assert fig is not None


# ---------------------------------------------------------------------------
# Interactive Plotting
# ---------------------------------------------------------------------------

class TestInteractivePlottingNumpyInput:
    def test_iplot_lead(self):
        from ecgdatakit.plotting import iplot_lead
        fig = iplot_lead(_sine_signal(), fs=500)
        assert fig is not None

    def test_iplot_lead_no_fs_raises(self):
        from ecgdatakit.plotting import iplot_lead
        with pytest.raises(TypeError, match="fs"):
            iplot_lead(_sine_signal())

    def test_iplot_spectrum(self):
        from ecgdatakit.plotting import iplot_spectrum
        fig = iplot_spectrum(_sine_signal(), fs=500)
        assert fig is not None


# ---------------------------------------------------------------------------
# Backward compatibility — Lead objects still work
# ---------------------------------------------------------------------------

class TestLeadBackwardCompat:
    """Verify existing Lead-based calls are unaffected."""

    def test_lowpass_with_lead(self):
        from ecgdatakit.processing.filters import lowpass
        lead = Lead(label="II", samples=_sine_signal(), sample_rate=500)
        result = lowpass(lead, cutoff=50)
        assert isinstance(result, Lead)
        assert result.label == "II"

    def test_plot_lead_with_lead(self):
        from ecgdatakit.plotting import plot_lead
        lead = Lead(label="II", samples=_sine_signal(), sample_rate=500, units="mV")
        fig = plot_lead(lead)
        assert fig is not None

    def test_fs_ignored_for_lead(self):
        """When a Lead is passed, fs is silently ignored."""
        from ecgdatakit.processing.filters import lowpass
        lead = Lead(label="II", samples=_sine_signal(), sample_rate=500)
        result = lowpass(lead, cutoff=50, fs=9999)  # fs should be ignored
        assert result.sample_rate == 500  # original sample_rate preserved
