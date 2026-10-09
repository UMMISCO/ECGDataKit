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
    return Lead(label=label, samples=signal.astype(np.float64), sampling_rate=fs)

def dominant_freq(lead):
    """Find the dominant frequency in a lead's signal."""
    n = len(lead.samples)
    yf = np.abs(np.fft.rfft(lead.samples))
    xf = np.fft.rfftfreq(n, d=1.0/lead.sampling_rate)
    return xf[np.argmax(yf[1:]) + 1]  # skip DC

class TestLowpass:
    def test_removes_high_frequency(self):
        lead = make_lead([10, 200], fs=500)
        result = lowpass(lead, cutoff=50)
        # After lowpass at 50 Hz, 200 Hz should be gone
        n = len(result.samples)
        yf = np.abs(np.fft.rfft(result.samples))
        xf = np.fft.rfftfreq(n, d=1.0/result.sampling_rate)
        # Power at 200 Hz should be < 1% of power at 10 Hz
        idx_10 = np.argmin(np.abs(xf - 10))
        idx_200 = np.argmin(np.abs(xf - 200))
        assert yf[idx_200] < yf[idx_10] * 0.01

    def test_preserves_metadata(self):
        lead = make_lead([10], fs=500, label="V1")
        result = lowpass(lead, cutoff=100)
        assert result.label == "V1"
        assert result.sampling_rate == 500
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
        xf = np.fft.rfftfreq(n, d=1.0/result.sampling_rate)
        idx_10 = np.argmin(np.abs(xf - 10))
        idx_50 = np.argmin(np.abs(xf - 50))
        assert yf[idx_50] < yf[idx_10] * 0.1

class TestRemoveBaseline:
    def test_removes_drift(self):
        t = np.arange(0, 5.0, 1.0/500)
        signal = np.sin(2 * np.pi * 10 * t) + 2.0 * np.sin(2 * np.pi * 0.1 * t)
        lead = Lead(label="II", samples=signal, sampling_rate=500)
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


class TestFilterValidation:
    def test_matches_scipy_reference(self):
        from scipy import signal as ss
        x = np.random.default_rng(0).standard_normal(2000)
        sos = ss.butter(4, [0.5, 40], btype="band", fs=500, output="sos")
        np.testing.assert_allclose(bandpass(x, 0.5, 40, fs=500).samples, ss.sosfiltfilt(sos, x))

    @pytest.mark.parametrize("order", [0, -1, 2.5])
    def test_bad_order(self, order):
        with pytest.raises(ValueError, match="order"):
            lowpass(make_lead([5]), 40, order=order)

    def test_nan_cutoff(self):
        with pytest.raises(ValueError, match="finite"):
            lowpass(make_lead([5]), float("nan"))

    def test_bad_quality(self):
        with pytest.raises(ValueError, match="quality"):
            notch(make_lead([5]), quality=0)

    def test_zero_sampling_rate(self):
        with pytest.raises(ValueError, match="sampling rate"):
            lowpass(np.zeros(100), 40, fs=0)

    def test_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            bandpass(np.zeros(10), 0.5, 40, fs=500)

    def test_nan_refused(self):
        lead = make_lead([5])
        lead.samples[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            bandpass(lead, 0.5, 40)

    def test_2d_refused(self):
        with pytest.raises(ValueError, match="1-D"):
            lowpass(np.zeros((12, 500)), 40, fs=500)

    def test_raw_offset_not_reintroduced(self):
        t = np.arange(2000) / 500
        lead = Lead(label="II", samples=np.sin(2 * np.pi * 5 * t) * 100, sampling_rate=500,
                    resolution=0.01, resolution_unit="mV", offset=5.0, is_raw=True)
        via_raw = highpass(lead, 0.5).to_physical().samples
        via_phys = highpass(lead.to_physical(), 0.5).samples
        np.testing.assert_allclose(via_raw, via_phys, atol=1e-9)


class TestPresetsLowRate:
    def test_diagnostic_at_180hz_lowers_cutoff(self):
        lead = make_lead([5, 70], fs=180, duration=10.0)
        with pytest.warns(UserWarning, match="81 Hz"):
            result = diagnostic_filter(lead)
        assert len(result.samples) == len(lead.samples)

    def test_notch_skipped_at_nyquist(self):
        lead = make_lead([5], fs=100, duration=10.0)
        with pytest.warns(UserWarning, match="notch at 50.0 Hz skipped"):
            monitoring_filter(lead)

    def test_notch_none(self):
        lead = make_lead([5, 50], fs=500, duration=4.0)
        result = diagnostic_filter(lead, notch_freq=None)
        # 50 Hz component kept without a notch
        assert np.abs(result.samples[500:-500]).max() > 1.5


class TestWarningLocation:
    def test_preset_warnings_point_at_caller(self):
        lead = make_lead([5], fs=100, duration=10.0)
        with pytest.warns(UserWarning) as record:
            diagnostic_filter(lead)
        assert len(record) == 2
        assert {w.filename for w in record} == {__file__}

    def test_scipy_install_hint_is_quoted(self, monkeypatch):
        import importlib
        from ecgdatakit.processing import _core

        def fail(name):
            raise ImportError("no scipy")

        monkeypatch.setattr(importlib, "import_module", fail)
        with pytest.raises(ImportError, match=r'pip install "ecgdatakit\[processing\]"'):
            _core.require_scipy("signal")
