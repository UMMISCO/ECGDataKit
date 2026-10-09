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


# ---------------------------------------------------------------------------
# Realistic synthetic ECG (P, Q, R, S, T) for accuracy and robustness tests
# ---------------------------------------------------------------------------

_WAVES = [(-0.20, 0.15, 0.025), (-0.03, -0.10, 0.010), (0.0, 1.0, 0.010),
          (0.03, -0.25, 0.010), (0.25, 0.30, 0.050)]


def synth_ecg(fs, duration=30.0, bpm=70, seed=0, rr_jitter=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    x = np.zeros_like(t)
    peaks = []
    pos = 0.6
    while pos < duration - 0.6:
        for off, amp, w in _WAVES:
            x += amp * np.exp(-0.5 * ((t - pos - off) / w) ** 2)
        peaks.append(int(round(pos * fs)))
        pos += 60.0 / bpm * (1 + rr_jitter * rng.uniform(-1, 1))
    x += 0.02 * rng.standard_normal(len(t))
    return x, np.array(peaks)


def match_count(truth, detected, tol):
    detected = np.asarray(detected)
    return sum(bool(np.any(np.abs(detected - p) <= tol)) for p in truth)


METHODS = ["shannon_energy", "pan_tompkins"]


class TestDetectorAccuracy:
    def test_default_method_is_pan_tompkins(self):
        x, _ = synth_ecg(500, duration=10)
        np.testing.assert_array_equal(
            detect_r_peaks(x, fs=500), detect_r_peaks(x, "pan_tompkins", fs=500)
        )

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("fs", [250, 360, 500, 1000])
    def test_sampling_rates(self, method, fs):
        x, truth = synth_ecg(fs, rr_jitter=0.15)
        det = detect_r_peaks(x, method, fs=fs)
        tol = int(round(0.010 * fs))
        assert len(det) == len(truth)
        assert match_count(truth, det, tol) == len(truth)

    @pytest.mark.parametrize("method", METHODS)
    def test_negative_polarity_located_on_dominant_deflection(self, method):
        fs = 500
        x, truth = synth_ecg(fs)
        det = detect_r_peaks(-x, method, fs=fs)
        assert len(det) == len(truth)
        assert np.max(np.abs(det - truth)) <= int(round(0.010 * fs))

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("amplitude", [10.0, 50.0])
    def test_single_artifact_does_not_lock_detector(self, method, amplitude):
        fs = 500
        x, truth = synth_ecg(fs, duration=60)
        x[2 * fs : 2 * fs + fs // 100] += amplitude
        det = detect_r_peaks(x, method, fs=fs)
        tol = int(round(0.050 * fs))
        assert match_count(truth, det, tol) >= len(truth) - 1
        # the artefact itself and its filter ringing may add two detections
        assert len(det) <= len(truth) + 2

    @pytest.mark.parametrize("method", METHODS)
    def test_recovers_after_amplitude_drop(self, method):
        fs = 500
        x, truth = synth_ecg(fs, duration=60)
        x[30 * fs :] *= 0.25
        det = detect_r_peaks(x, method, fs=fs)
        assert match_count(truth, det, int(round(0.050 * fs))) >= 0.95 * len(truth)

    @pytest.mark.parametrize("method", METHODS)
    def test_output_sorted_unique(self, method):
        x, _ = synth_ecg(360, rr_jitter=0.2)
        det = detect_r_peaks(x, method, fs=360)
        assert np.all(np.diff(det) > 0)


class TestValidation:
    @pytest.mark.parametrize("method", METHODS)
    def test_nan_raises(self, method):
        x, _ = synth_ecg(500, duration=5)
        x[100] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            detect_r_peaks(x, method, fs=500)

    @pytest.mark.parametrize("n", [0, 10, 499])
    def test_short_signal_raises(self, n):
        with pytest.raises(ValueError, match="too short"):
            detect_r_peaks(np.zeros(n), fs=500)

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            detect_r_peaks(np.zeros((2, 5000)), fs=500)

    def test_flat_signal_no_peaks_and_nan_heart_rate(self):
        flat = np.zeros(5000)
        assert len(detect_r_peaks(flat, fs=500)) == 0
        assert np.isnan(heart_rate(flat, fs=500))

    def test_unsorted_peaks_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            heart_rate(np.zeros(5000), np.array([500, 100, 900]), fs=500)

    def test_list_peaks_accepted(self):
        rr = rr_intervals(np.zeros(5000), [100, 600, 1100], fs=500)
        np.testing.assert_allclose(rr, [1000.0, 1000.0])


class TestFlatAndPadding:
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("value", [0.0, 3.0, 1024.0])
    def test_constant_signal_has_no_peaks(self, method, value):
        assert len(detect_r_peaks(np.full(5000, value), method, fs=500)) == 0

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("pad", [0.0, 0.7, -2.0])
    def test_ecg_followed_by_flat_padding(self, method, pad):
        # 2.5 s of ECG then 7.5 s of constant padding, as some exports store
        fs = 500
        x, truth = synth_ecg(fs, duration=10)
        keep = int(2.5 * fs)
        x[keep:] = pad
        det = detect_r_peaks(x, method, fs=fs)
        expected = truth[truth < keep - int(0.1 * fs)]
        assert len(det) == len(expected)
        # placed on R, not on the S wave 30 ms later
        assert np.max(np.abs(det - expected)) <= int(round(0.005 * fs))

    @pytest.mark.parametrize("method", METHODS)
    def test_raw_counts_with_offset_and_zero_padding(self, method):
        fs = 500
        x, truth = synth_ecg(fs, duration=10)
        keep = int(2.5 * fs)
        counts = 1024.0 + 200.0 * x
        counts[keep:] = 0.0
        det = detect_r_peaks(counts, method, fs=fs)
        np.testing.assert_array_equal(det, truth[truth < keep - int(0.1 * fs)])

    def test_flat_run_inside_recording(self):
        fs = 500
        x, truth = synth_ecg(fs, duration=30)
        x[10 * fs : 14 * fs] = 0.0
        det = detect_r_peaks(x, fs=fs)
        expected = truth[(truth < 10 * fs - 50) | (truth > 14 * fs + 50)]
        assert match_count(expected, det, 5) == len(expected)
        assert not np.any((det > 10 * fs + 10) & (det < 14 * fs - 10))

    def test_square_wave_artifact_limitation(self):
        # Lead-off square waves are not recognised: Pan-Tompkins still finds
        # every beat around the artefact but reports its edges as beats.
        fs = 500
        rng = np.random.default_rng(1)
        x, truth = synth_ecg(fs, duration=20)
        n = 4 * fs
        x[8 * fs : 12 * fs] = (np.where((np.arange(n) // (fs // 2)) % 2, 1.0, -1.0)
                               + 0.01 * rng.standard_normal(n))
        det = detect_r_peaks(x, "pan_tompkins", fs=fs)
        outside = truth[(truth < 7.8 * fs) | (truth > 12.2 * fs)]
        assert match_count(outside, det, 5) == len(outside)


def synth_ecg_waves(fs, waves, duration=60.0, bpm=75, drop_every=0, seed=0):
    """Synthetic ECG with custom waves; every *drop_every*-th beat is omitted."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    x = 0.02 * rng.standard_normal(len(t))
    peaks = []
    pos, k = 0.6, 0
    while pos < duration - 0.6:
        k += 1
        if not (drop_every and k % drop_every == 0):
            for off, amp, w in waves:
                x += amp * np.exp(-0.5 * ((t - pos - off) / w) ** 2)
            peaks.append(int(round(pos * fs)))
        pos += 60.0 / bpm
    return x, np.array(peaks)


class TestPanTompkinsRegressions:
    @pytest.mark.parametrize("t_amp, t_width", [(0.5, 0.03), (0.8, 0.04), (1.2, 0.05)])
    @pytest.mark.parametrize("drop_every", [3, 5])
    def test_large_t_waves_not_counted_after_pauses(self, t_amp, t_width, drop_every):
        # Dropped beats trigger searchback; the skipped T waves must not be
        # taken as beats.
        fs = 500
        waves = [(-0.20, 0.15, 0.025), (-0.03, -0.10, 0.010), (0.0, 1.0, 0.010),
                 (0.03, -0.25, 0.010), (0.28, t_amp, t_width)]
        x, truth = synth_ecg_waves(fs, waves, drop_every=drop_every)
        det = detect_r_peaks(x, "pan_tompkins", fs=fs)
        tol = int(round(0.025 * fs))
        assert match_count(truth, det, tol) == len(truth)
        assert len(det) == len(truth)

    @pytest.mark.parametrize("method", METHODS)
    def test_amplitude_step_up_x3(self, method):
        fs = 500
        x, truth = synth_ecg(fs, duration=120)
        x[60 * fs :] *= 3.0
        det = detect_r_peaks(x, method, fs=fs)
        found = match_count(truth, det, int(round(0.050 * fs)))
        assert len(truth) - found <= 5
        assert len(det) - found <= 2


class TestFlatGaps:
    def test_rr_spans_a_skipped_flat_run(self):
        # 10 s ECG, 5 s flat, 10 s ECG: no beats in the gap, and the RR
        # interval across it covers the whole gap.
        fs = 500
        a, truth_a = synth_ecg(fs, duration=10)
        b, truth_b = synth_ecg(fs, duration=10, seed=1)
        x = np.concatenate([a, np.zeros(5 * fs), b])
        truth = np.concatenate([truth_a, truth_b + 15 * fs])
        det = detect_r_peaks(x, fs=fs)
        assert match_count(truth, det, 5) == len(truth) == len(det)
        rr = rr_intervals(x, det, fs=fs)
        gap_rr = (truth_b[0] + 15 * fs - truth_a[-1]) * 1000.0 / fs
        assert rr.max() == pytest.approx(gap_rr, abs=10.0)
        assert rr.max() > 5000.0

    def test_short_stretch_between_flat_runs_not_analysed(self):
        fs = 500
        x, _ = synth_ecg(fs, duration=10)
        y = np.concatenate([np.zeros(3 * fs), x[: int(0.8 * fs)], np.zeros(3 * fs)])
        assert len(detect_r_peaks(y, fs=fs)) == 0
