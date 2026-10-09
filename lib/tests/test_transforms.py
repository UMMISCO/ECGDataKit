import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.transforms import power_spectrum, fft, segment_beats, average_beat

def make_lead(freq=10, fs=500, duration=2.0, label="II"):
    t = np.arange(0, duration, 1.0 / fs)
    signal = np.sin(2 * np.pi * freq * t).astype(np.float64)
    return Lead(label=label, samples=signal, sampling_rate=fs)

class TestPowerSpectrum:
    def test_peak_at_signal_frequency(self):
        lead = make_lead(freq=10, fs=500, duration=4.0)
        freqs, psd = power_spectrum(lead)
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 10.0) < 2.0

    def test_returns_arrays(self):
        lead = make_lead()
        freqs, psd = power_spectrum(lead)
        assert len(freqs) == len(psd)
        assert len(freqs) > 0

class TestFFT:
    def test_dominant_frequency(self):
        lead = make_lead(freq=25, fs=500, duration=2.0)
        freqs, mags = fft(lead)
        # Dominant frequency should be 25 Hz
        peak_freq = freqs[np.argmax(mags[1:]) + 1]  # skip DC
        assert abs(peak_freq - 25.0) < 1.0

class TestSegmentBeats:
    def test_correct_number_of_segments(self):
        # Create a lead with known peaks
        signal = np.zeros(5000, dtype=np.float64)
        peaks = np.array([500, 1500, 2500, 3500, 4500], dtype=np.intp)
        for p in peaks:
            signal[p] = 10.0  # Sharp peak
        lead = Lead(label="II", samples=signal, sampling_rate=500)
        beats = segment_beats(lead, peaks=peaks, before=0.1, after=0.2)
        # First and last peaks may be skipped if too close to edges
        assert len(beats) >= 3

    def test_beat_labels(self):
        signal = np.zeros(3000, dtype=np.float64)
        peaks = np.array([500, 1500, 2500], dtype=np.intp)
        lead = Lead(label="V1", samples=signal, sampling_rate=500)
        beats = segment_beats(lead, peaks=peaks, before=0.1, after=0.2)
        for beat in beats:
            assert beat.label.startswith("V1_beat_")

class TestAverageBeat:
    def test_returns_single_lead(self):
        signal = np.zeros(3000, dtype=np.float64)
        peaks = np.array([500, 1500, 2500], dtype=np.intp)
        for p in peaks:
            signal[p] = 10.0
        lead = Lead(label="II", samples=signal, sampling_rate=500)
        avg = average_beat(lead, peaks=peaks, before=0.1, after=0.2)
        assert avg.label == "II_avg"
        assert len(avg.samples) > 0


class TestFFTScaling:
    def test_dc_not_doubled(self):
        _, mags = fft(np.full(1000, 3.0), fs=500)
        assert mags[0] == pytest.approx(3.0)

    def test_nyquist_not_doubled(self):
        n = np.arange(1000)
        _, mags = fft(np.cos(np.pi * n), fs=500)
        assert mags[-1] == pytest.approx(1.0)

    def test_interior_amplitude(self):
        freqs, mags = fft(make_lead(freq=10, fs=500, duration=2.0))
        assert mags[np.argmin(np.abs(freqs - 10))] == pytest.approx(1.0, abs=1e-9)

    def test_empty_refused(self):
        with pytest.raises(ValueError, match="no samples"):
            fft(np.array([]), fs=500)


class TestPowerSpectrumMethods:
    def test_periodogram(self):
        from scipy import signal as ss
        lead = make_lead(freq=10, fs=500, duration=2.0)
        freqs, psd = power_spectrum(lead, method="periodogram")
        ref_f, ref_p = ss.periodogram(lead.samples, fs=500)
        np.testing.assert_allclose(psd, ref_p)
        np.testing.assert_allclose(freqs, ref_f)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            power_spectrum(make_lead(), method="bogus")


class TestBeatsEdgeCases:
    def test_whole_float_peaks(self):
        beats = segment_beats(make_lead(duration=4.0), peaks=np.array([500.0, 1000.0]))
        assert len(beats) == 2

    def test_fractional_peaks_refused(self):
        with pytest.raises(ValueError, match="integer"):
            segment_beats(make_lead(duration=4.0), peaks=np.array([500.5]))

    def test_average_beat_without_beats_raises(self):
        with pytest.raises(ValueError, match="no complete beat"):
            average_beat(make_lead(duration=4.0), peaks=np.array([5]))
