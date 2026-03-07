import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.peaks import detect_r_peaks, heart_rate, rr_intervals, instantaneous_heart_rate

def make_ecg_lead(fs=500, duration=10.0, bpm=72):
    """Create a synthetic ECG-like signal with R-peaks at known positions."""
    t = np.arange(0, duration, 1.0 / fs)
    signal = np.zeros_like(t)
    rr_s = 60.0 / bpm  # RR interval in seconds
    peak_positions = []
    pos = rr_s  # first peak
    while pos < duration - rr_s:
        idx = int(pos * fs)
        peak_positions.append(idx)
        # Sharp Gaussian pulse (QRS-like)
        sigma = 0.01 * fs  # ~10 ms width
        gaussian = np.exp(-0.5 * ((np.arange(len(t)) - idx) / sigma) ** 2)
        signal += gaussian
        pos += rr_s
    # Add small baseline wander
    signal += 0.05 * np.sin(2 * np.pi * 0.3 * t)
    lead = Lead(label="II", samples=signal.astype(np.float64), sampling_rate=fs)
    return lead, np.array(peak_positions, dtype=np.intp)

class TestDetectRPeaks:
    def test_finds_correct_number_of_peaks(self):
        lead, expected_peaks = make_ecg_lead(bpm=72, duration=10.0)
        detected = detect_r_peaks(lead)
        # Should find approximately the right number (within ±2)
        assert abs(len(detected) - len(expected_peaks)) <= 2

    def test_peaks_near_expected_positions(self):
        lead, expected_peaks = make_ecg_lead(bpm=60, duration=10.0)
        detected = detect_r_peaks(lead)
        # Each detected peak should be within ±25 samples of an expected peak
        tolerance = 25
        for d in detected:
            min_dist = np.min(np.abs(expected_peaks - d))
            assert min_dist < tolerance, f"Peak at {d} not near any expected peak"

    def test_unknown_method_raises(self):
        lead, _ = make_ecg_lead()
        with pytest.raises(ValueError, match="Unknown method"):
            detect_r_peaks(lead, method="unknown")

class TestHeartRate:
    def test_known_bpm(self):
        lead, expected = make_ecg_lead(bpm=72, duration=20.0)
        hr = heart_rate(lead, peaks=expected)
        assert abs(hr - 72.0) < 2.0

class TestRRIntervals:
    def test_returns_correct_intervals(self):
        lead, peaks = make_ecg_lead(bpm=60, duration=10.0)
        rr = rr_intervals(lead, peaks=peaks)
        # At 60 bpm, RR should be ~1000 ms
        assert len(rr) == len(peaks) - 1
        assert np.all(np.abs(rr - 1000.0) < 5.0)  # within 5 ms

    def test_empty_with_fewer_than_2_peaks(self):
        lead = Lead(label="II", samples=np.zeros(100, dtype=np.float64), sampling_rate=500)
        rr = rr_intervals(lead, peaks=np.array([50], dtype=np.intp))
        assert len(rr) == 0

class TestInstantaneousHeartRate:
    def test_returns_bpm_per_beat(self):
        lead, peaks = make_ecg_lead(bpm=72, duration=10.0)
        ihr = instantaneous_heart_rate(lead, peaks=peaks)
        assert len(ihr) == len(peaks) - 1
        # Each should be approximately 72 bpm
        assert np.all(np.abs(ihr - 72.0) < 3.0)
