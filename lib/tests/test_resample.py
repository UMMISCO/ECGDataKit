import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.resample import resample

def make_lead(freq=10, fs=500, duration=2.0):
    t = np.arange(0, duration, 1.0/fs)
    signal = np.sin(2 * np.pi * freq * t)
    return Lead(label="II", samples=signal.astype(np.float64), sampling_rate=fs)

class TestResample:
    def test_downsample_halves_length(self):
        lead = make_lead(fs=500, duration=2.0)
        result = resample(lead, 250)
        assert result.sampling_rate == 250
        assert abs(len(result.samples) - 500) <= 2  # 250 Hz * 2 s

    def test_upsample_doubles_length(self):
        lead = make_lead(fs=250, duration=2.0)
        result = resample(lead, 500)
        assert result.sampling_rate == 500
        assert abs(len(result.samples) - 1000) <= 2

    def test_preserves_frequency_content(self):
        lead = make_lead(freq=10, fs=500, duration=2.0)
        result = resample(lead, 250)
        n = len(result.samples)
        yf = np.abs(np.fft.rfft(result.samples))
        xf = np.fft.rfftfreq(n, d=1.0/250)
        dominant = xf[np.argmax(yf[1:]) + 1]
        assert abs(dominant - 10.0) < 1.0

    def test_same_rate_copies(self):
        lead = make_lead(fs=500)
        result = resample(lead, 500)
        assert result.sampling_rate == 500
        np.testing.assert_array_equal(result.samples, lead.samples)

    def test_invalid_rate_raises(self):
        lead = make_lead()
        with pytest.raises(ValueError):
            resample(lead, 0)

    def test_preserves_metadata(self):
        lead = Lead(label="V3", samples=np.ones(100, dtype=np.float64), sampling_rate=500, units="mV")
        result = resample(lead, 250)
        assert result.label == "V3"
        assert result.units == "mV"


class TestResampleEdgesAndRates:
    def test_dc_level_has_no_edge_ringing(self):
        lead = Lead(label="II", samples=np.full(5000, 1024.0), sampling_rate=500)
        result = resample(lead, 360)
        # Zero padding used to give 881 at the first sample
        np.testing.assert_allclose(result.samples, 1024.0, rtol=2e-4)

    def test_whole_float_target(self):
        result = resample(make_lead(fs=500), 250.0)
        assert result.sampling_rate == 250 and isinstance(result.sampling_rate, int)

    def test_fractional_target_refused(self):
        with pytest.raises(ValueError, match="whole number"):
            resample(make_lead(fs=500), 250.5)

    def test_float_source_rate(self):
        result = resample(np.zeros(999), 500, fs=499.5)
        assert len(result.samples) == 1000

    def test_zero_source_rate(self):
        with pytest.raises(ValueError, match="sampling rate"):
            resample(np.zeros(100), 250, fs=0)

    def test_2d_refused(self):
        with pytest.raises(ValueError, match="1-D"):
            resample(np.zeros((12, 5000)), 250, fs=500)

    def test_nan_refused(self):
        with pytest.raises(ValueError, match="NaN"):
            resample(np.array([0.0, np.nan, 1.0, 2.0]), 250, fs=500)
