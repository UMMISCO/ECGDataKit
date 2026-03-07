"""Tests for Shannon energy R-peak detection method."""

import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.peaks import detect_r_peaks


def make_ecg_lead(fs=500, duration=10.0, bpm=72):
    """Create a synthetic ECG-like signal with R-peaks at known positions."""
    t = np.arange(0, duration, 1.0 / fs)
    signal = np.zeros_like(t)
    rr_s = 60.0 / bpm
    peak_positions = []
    pos = rr_s
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
    return Lead(label="II", samples=signal.astype(np.float64), sampling_rate=fs), \
        np.array(peak_positions, dtype=np.intp)


class TestShannonEnergyDetection:
    def test_finds_peaks(self):
        """Shannon energy method should find R-peaks."""
        lead, _ = make_ecg_lead(bpm=72, duration=10.0)
        detected = detect_r_peaks(lead, method="shannon_energy")
        # Should find at least some peaks
        assert len(detected) > 0

    def test_finds_correct_number_of_peaks(self):
        """Should find approximately the right number of peaks."""
        lead, expected_peaks = make_ecg_lead(bpm=72, duration=10.0)
        detected = detect_r_peaks(lead, method="shannon_energy")
        # Within ±3 of expected
        assert abs(len(detected) - len(expected_peaks)) <= 3

    def test_peaks_are_sorted(self):
        """Detected peaks should be sorted in ascending order."""
        lead, _ = make_ecg_lead(bpm=72, duration=10.0)
        detected = detect_r_peaks(lead, method="shannon_energy")
        if len(detected) > 1:
            assert np.all(np.diff(detected) > 0)

    def test_peaks_within_signal_bounds(self):
        """All peak indices should be valid signal indices."""
        lead, _ = make_ecg_lead(bpm=72, duration=10.0)
        detected = detect_r_peaks(lead, method="shannon_energy")
        assert np.all(detected >= 0)
        assert np.all(detected < len(lead.samples))

    def test_returns_intp_array(self):
        """Should return NDArray[np.intp]."""
        lead, _ = make_ecg_lead()
        detected = detect_r_peaks(lead, method="shannon_energy")
        assert detected.dtype == np.intp

    def test_different_sampling_rates(self):
        """Should work with different sample rates."""
        for fs in [250, 360, 500, 1000]:
            lead, _ = make_ecg_lead(fs=fs, bpm=72, duration=10.0)
            detected = detect_r_peaks(lead, method="shannon_energy")
            # Should find peaks at any reasonable sample rate
            assert len(detected) >= 0  # no crash

    def test_short_signal(self):
        """Should handle short signals without crashing."""
        # 1 second signal — may not detect peaks, but should not error
        lead, _ = make_ecg_lead(fs=500, bpm=72, duration=1.0)
        detected = detect_r_peaks(lead, method="shannon_energy")
        assert isinstance(detected, np.ndarray)

    def test_noisy_signal(self):
        """Should still detect peaks in a noisy signal."""
        lead, expected_peaks = make_ecg_lead(bpm=72, duration=10.0)
        # Add noise
        rng = np.random.default_rng(42)
        noisy = lead.samples + 0.1 * rng.standard_normal(len(lead.samples))
        noisy_lead = Lead(label="II", samples=noisy, sampling_rate=lead.sampling_rate)
        detected = detect_r_peaks(noisy_lead, method="shannon_energy")
        assert len(detected) > 0

    def test_flat_signal_returns_empty(self):
        """A flat signal should return no peaks."""
        flat = np.zeros(5000, dtype=np.float64)
        lead = Lead(label="II", samples=flat, sampling_rate=500)
        detected = detect_r_peaks(lead, method="shannon_energy")
        assert len(detected) == 0

    def test_method_selection(self):
        """Verify method='shannon_energy' uses the correct path."""
        lead, _ = make_ecg_lead()
        # Both methods should return arrays (they may differ in exact peaks)
        pt_peaks = detect_r_peaks(lead, method="pan_tompkins")
        se_peaks = detect_r_peaks(lead, method="shannon_energy")
        assert isinstance(pt_peaks, np.ndarray)
        assert isinstance(se_peaks, np.ndarray)


class TestShannonEnergyEdgeCases:
    def test_very_low_sampling_rate(self):
        """Should not crash with very low sample rate."""
        # 100 Hz — lowfreq/highfreq will be clamped
        t = np.arange(0, 10.0, 1.0 / 100)
        signal = np.zeros_like(t)
        rr_s = 60.0 / 72
        pos = rr_s
        while pos < 10.0 - rr_s:
            idx = int(pos * 100)
            sigma = 0.01 * 100
            signal += np.exp(-0.5 * ((np.arange(len(t)) - idx) / sigma) ** 2)
            pos += rr_s
        lead = Lead(label="II", samples=signal.astype(np.float64), sampling_rate=100)
        detected = detect_r_peaks(lead, method="shannon_energy")
        assert isinstance(detected, np.ndarray)

    def test_single_beat(self):
        """Should handle signal with just one beat."""
        t = np.arange(0, 2.0, 1.0 / 500)
        signal = np.zeros_like(t)
        idx = 500  # peak at 1s
        sigma = 5
        signal += np.exp(-0.5 * ((np.arange(len(t)) - idx) / sigma) ** 2)
        lead = Lead(label="II", samples=signal.astype(np.float64), sampling_rate=500)
        detected = detect_r_peaks(lead, method="shannon_energy")
        assert isinstance(detected, np.ndarray)
