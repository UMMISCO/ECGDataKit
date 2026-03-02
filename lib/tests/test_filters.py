import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.filters import (
    lowpass, highpass, bandpass, notch, remove_baseline,
    diagnostic_filter, monitoring_filter,
)

def make_lead(freqs, fs=500, duration=2.0, label="II"):
    """Create a Lead with a sum of sine waves at given frequencies."""
    t = np.arange(0, duration, 1.0/fs)
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return Lead(label=label, samples=signal.astype(np.float64), sample_rate=fs)

def dominant_freq(lead):
    """Find the dominant frequency in a lead's signal."""
    n = len(lead.samples)
    yf = np.abs(np.fft.rfft(lead.samples))
    xf = np.fft.rfftfreq(n, d=1.0/lead.sample_rate)
    return xf[np.argmax(yf[1:]) + 1]  # skip DC

class TestLowpass:
    def test_removes_high_frequency(self):
        lead = make_lead([10, 200], fs=500)
        result = lowpass(lead, cutoff=50)
        # After lowpass at 50 Hz, 200 Hz should be gone
        n = len(result.samples)
        yf = np.abs(np.fft.rfft(result.samples))
        xf = np.fft.rfftfreq(n, d=1.0/result.sample_rate)
        # Power at 200 Hz should be < 1% of power at 10 Hz
        idx_10 = np.argmin(np.abs(xf - 10))
        idx_200 = np.argmin(np.abs(xf - 200))
        assert yf[idx_200] < yf[idx_10] * 0.01

    def test_preserves_metadata(self):
        lead = make_lead([10], fs=500, label="V1")
        result = lowpass(lead, cutoff=100)
        assert result.label == "V1"
        assert result.sample_rate == 500
        assert len(result.samples) == len(lead.samples)

    def test_returns_new_lead(self):
        lead = make_lead([10])
        result = lowpass(lead, cutoff=100)
        assert result is not lead

class TestHighpass:
    def test_removes_dc_and_drift(self):
        lead = make_lead([0.1, 10], fs=500, duration=5.0)
        result = highpass(lead, cutoff=1.0)
        # Dominant frequency should be 10 Hz
        assert abs(dominant_freq(result) - 10.0) < 1.0

class TestBandpass:
    def test_passes_target_removes_extremes(self):
        lead = make_lead([0.1, 10, 200], fs=500, duration=5.0)
        result = bandpass(lead, low=1.0, high=50.0)
        assert abs(dominant_freq(result) - 10.0) < 1.0

    def test_invalid_range_raises(self):
        lead = make_lead([10])
        with pytest.raises(ValueError):
            bandpass(lead, low=50, high=10)

class TestNotch:
    def test_removes_50hz(self):
        lead = make_lead([10, 50], fs=500)
        result = notch(lead, freq=50.0)
        n = len(result.samples)
        yf = np.abs(np.fft.rfft(result.samples))
        xf = np.fft.rfftfreq(n, d=1.0/result.sample_rate)
        idx_10 = np.argmin(np.abs(xf - 10))
        idx_50 = np.argmin(np.abs(xf - 50))
        assert yf[idx_50] < yf[idx_10] * 0.1

class TestRemoveBaseline:
    def test_removes_drift(self):
        t = np.arange(0, 5.0, 1.0/500)
        signal = np.sin(2 * np.pi * 10 * t) + 2.0 * np.sin(2 * np.pi * 0.1 * t)
        lead = Lead(label="II", samples=signal, sample_rate=500)
        result = remove_baseline(lead)
        assert abs(dominant_freq(result) - 10.0) < 1.0

class TestCutoffValidation:
    def test_cutoff_above_nyquist_raises(self):
        lead = make_lead([10], fs=500)
        with pytest.raises(ValueError, match="Nyquist"):
            lowpass(lead, cutoff=260)

    def test_negative_cutoff_raises(self):
        lead = make_lead([10], fs=500)
        with pytest.raises(ValueError):
            lowpass(lead, cutoff=-1)

class TestPresets:
    def test_diagnostic_filter_runs(self):
        lead = make_lead([10], fs=500, duration=2.0)
        result = diagnostic_filter(lead)
        assert len(result.samples) == len(lead.samples)

    def test_monitoring_filter_runs(self):
        lead = make_lead([10], fs=500, duration=2.0)
        result = monitoring_filter(lead)
        assert len(result.samples) == len(lead.samples)
