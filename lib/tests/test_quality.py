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
    return Lead(label="II", samples=signal.astype(np.float64), sampling_rate=fs)

def make_noisy_signal(fs=500, duration=10.0):
    """Pure random noise signal."""
    np.random.seed(123)
    signal = np.random.randn(int(fs * duration))
    return Lead(label="II", samples=signal.astype(np.float64), sampling_rate=fs)

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


_WAVES = [(-0.20, 0.15, 0.025), (-0.03, -0.10, 0.010), (0.0, 1.0, 0.010),
          (0.03, -0.25, 0.010), (0.25, 0.30, 0.050)]


def synth_ecg(fs, duration=10.0, bpm=70):
    t = np.arange(int(duration * fs)) / fs
    x = np.zeros_like(t)
    pos = 0.6
    while pos < duration - 0.6:
        for off, amp, w in _WAVES:
            x += amp * np.exp(-0.5 * ((t - pos - off) / w) ** 2)
        pos += 60.0 / bpm
    return x


class TestRateIndependence:
    @pytest.mark.parametrize("fs", [250, 360, 500, 1000])
    def test_baseline_wander_detected_at_all_rates(self, fs):
        from ecgdatakit.processing.quality import _baseline_sqi, _power_ratio_sqi
        x = synth_ecg(fs)
        t = np.arange(len(x)) / fs
        wander = x + 2.0 * np.sin(2 * np.pi * 0.3 * t)
        assert _baseline_sqi(x, fs) > 0.7
        assert _baseline_sqi(wander, fs) < 0.1
        assert _power_ratio_sqi(x, fs) == 1.0
        assert _power_ratio_sqi(wander, fs) < 0.2
        assert classify_quality(x, fs=fs) == "excellent"
        assert classify_quality(wander, fs=fs) == "unacceptable"


class TestSnrLimits:
    @pytest.mark.parametrize("fs", [128, 200])
    def test_nan_at_or_below_200_hz(self, fs):
        assert np.isnan(snr_estimate(synth_ecg(fs), fs=fs))

    def test_white_noise_level(self):
        fs = 500
        x = synth_ecg(fs)
        rng = np.random.default_rng(0)
        noisy = x + 0.05 * rng.standard_normal(len(x))
        # noise density 0.05**2 / 250 Hz over the 39 Hz ECG band
        expected = 10 * np.log10(np.var(x) / (0.05 ** 2 * 39.0 / 250.0))
        assert snr_estimate(noisy, fs=fs) == pytest.approx(expected, abs=1.5)


class TestQualityValidation:
    def test_nan_raises(self):
        x = synth_ecg(500)
        x[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            signal_quality_index(x, fs=500)

    def test_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            signal_quality_index(np.zeros(100), fs=500)
