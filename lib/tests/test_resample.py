import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.resample import resample

def make_lead(freq=10, fs=500, duration=2.0):
    t = np.arange(0, duration, 1.0/fs)
    signal = np.sin(2 * np.pi * freq * t)
    return Lead(label="II", samples=signal.astype(np.float64), sample_rate=fs)

class TestResample:
    def test_downsample_halves_length(self):
        lead = make_lead(fs=500, duration=2.0)
        result = resample(lead, 250)
        assert result.sample_rate == 250
        assert abs(len(result.samples) - 500) <= 2  # 250 Hz * 2 s

    def test_upsample_doubles_length(self):
        lead = make_lead(fs=250, duration=2.0)
        result = resample(lead, 500)
        assert result.sample_rate == 500
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
        assert result.sample_rate == 500
        np.testing.assert_array_equal(result.samples, lead.samples)

    def test_invalid_rate_raises(self):
        lead = make_lead()
        with pytest.raises(ValueError):
            resample(lead, 0)

    def test_preserves_metadata(self):
        lead = Lead(label="V3", samples=np.ones(100, dtype=np.float64), sample_rate=500, units="mV")
        result = resample(lead, 250)
        assert result.label == "V3"
        assert result.units == "mV"
