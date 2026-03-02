import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.quality import signal_quality_index, classify_quality, snr_estimate

def make_clean_ecg(fs=500, duration=10.0, bpm=72):
    """Synthetic clean ECG-like signal."""
    t = np.arange(0, duration, 1.0 / fs)
    signal = np.zeros_like(t)
    rr_s = 60.0 / bpm
    pos = rr_s
    while pos < duration - rr_s:
        idx = int(pos * fs)
        sigma = 0.01 * fs
        signal += np.exp(-0.5 * ((np.arange(len(t)) - idx) / sigma) ** 2)
        pos += rr_s
    return Lead(label="II", samples=signal.astype(np.float64), sample_rate=fs)

def make_noisy_signal(fs=500, duration=10.0):
    """Pure random noise signal."""
    np.random.seed(123)
    signal = np.random.randn(int(fs * duration))
    return Lead(label="II", samples=signal.astype(np.float64), sample_rate=fs)

class TestSignalQualityIndex:
    def test_clean_ecg_high_score(self):
        lead = make_clean_ecg()
        sqi = signal_quality_index(lead)
        assert sqi > 0.4  # Clean synthetic should score reasonably

    def test_noise_low_score(self):
        lead = make_noisy_signal()
        sqi = signal_quality_index(lead)
        assert sqi < 0.7  # Pure noise should score lower

    def test_range(self):
        lead = make_clean_ecg()
        sqi = signal_quality_index(lead)
        assert 0.0 <= sqi <= 1.0

class TestClassifyQuality:
    def test_returns_valid_category(self):
        lead = make_clean_ecg()
        result = classify_quality(lead)
        assert result in ("excellent", "acceptable", "unacceptable")

class TestSNREstimate:
    def test_clean_signal_higher_snr(self):
        clean = make_clean_ecg()
        noisy = make_noisy_signal()
        snr_clean = snr_estimate(clean)
        snr_noisy = snr_estimate(noisy)
        assert snr_clean > snr_noisy

    def test_returns_float(self):
        lead = make_clean_ecg()
        result = snr_estimate(lead)
        assert isinstance(result, float)
