"""Tests for ecgdatakit.processing.clean module."""

import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.clean import clean_ecg


def make_noisy_ecg(fs=500, duration=10.0, bpm=72, noise_level=0.3):
    """Create a synthetic ECG-like signal with added noise."""
    t = np.arange(0, duration, 1.0 / fs)
    signal = np.zeros_like(t)
    rr_s = 60.0 / bpm
    pos = rr_s
    while pos < duration - rr_s:
        idx = int(pos * fs)
        sigma = 0.01 * fs
        signal += np.exp(-0.5 * ((np.arange(len(t)) - idx) / sigma) ** 2)
        pos += rr_s
    signal += noise_level * np.sin(2 * np.pi * 50 * t)
    signal += 0.1 * np.sin(2 * np.pi * 0.3 * t)
    rng = np.random.default_rng(42)
    signal += 0.05 * rng.standard_normal(len(signal))
    return Lead(label="II", samples=signal.astype(np.float64), sampling_rate=fs)


class TestCleanECGDefault:
    def test_returns_lead(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="default")
        assert isinstance(result, Lead)

    def test_preserves_sampling_rate(self):
        lead = make_noisy_ecg(fs=250)
        result = clean_ecg(lead, method="default")
        assert result.sampling_rate == 250

    def test_preserves_label(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="default")
        assert result.label == "II"

    def test_preserves_length(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="default")
        assert len(result.samples) == len(lead.samples)

    def test_does_not_modify_original(self):
        lead = make_noisy_ecg()
        original_samples = lead.samples.copy()
        _ = clean_ecg(lead, method="default")
        np.testing.assert_array_equal(lead.samples, original_samples)

    def test_reduces_50hz_noise(self):
        """50 Hz interference should be attenuated."""
        lead = make_noisy_ecg(noise_level=0.5)
        result = clean_ecg(lead, method="default")
        from scipy.signal import welch
        _, pxx_before = welch(lead.samples, fs=lead.sampling_rate, nperseg=1024)
        _, pxx_after = welch(result.samples, fs=result.sampling_rate, nperseg=1024)
        freq_bins = np.arange(len(pxx_before)) * (lead.sampling_rate / 2) / len(pxx_before)
        idx_50 = np.argmin(np.abs(freq_bins - 50))
        assert pxx_after[idx_50] < pxx_before[idx_50]

    def test_output_is_float64(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="default")
        assert result.samples.dtype == np.float64


class TestCleanECGValidation:
    def test_unknown_method_raises(self):
        lead = make_noisy_ecg()
        with pytest.raises(ValueError, match="Unknown method"):
            clean_ecg(lead, method="invalid_method")

    def test_default_method(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead)
        assert isinstance(result, Lead)
        assert len(result.samples) == len(lead.samples)


class TestCleanECGBioSPPy:
    @pytest.fixture(autouse=True)
    def skip_if_no_biosppy(self):
        pytest.importorskip("biosppy")

    def test_returns_lead(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="biosppy")
        assert isinstance(result, Lead)

    def test_preserves_length(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="biosppy")
        assert len(result.samples) == len(lead.samples)


class TestCleanECGNeuroKit2:
    @pytest.fixture(autouse=True)
    def skip_if_no_neurokit2(self):
        pytest.importorskip("neurokit2")

    def test_returns_lead(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="neurokit2")
        assert isinstance(result, Lead)

    def test_preserves_length(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="neurokit2")
        assert len(result.samples) == len(lead.samples)


class TestCleanECGCombined:
    @pytest.fixture(autouse=True)
    def skip_if_missing_deps(self):
        pytest.importorskip("biosppy")
        pytest.importorskip("neurokit2")

    def test_returns_lead(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="combined")
        assert isinstance(result, Lead)

    def test_preserves_label(self):
        lead = make_noisy_ecg()
        result = clean_ecg(lead, method="combined")
        assert result.label == "II"


class TestCleanECGDeepFADE:
    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_deepfade_method_callable(self):
        """clean_ecg with method='deepfade' should be callable."""
        from ecgdatakit.processing.clean import _DEEPFADE_WEIGHTS
        lead = make_noisy_ecg(duration=10.0)
        if not _DEEPFADE_WEIGHTS.exists():
            pytest.skip("weights file not found")
        with pytest.warns(UserWarning, match="experimental"):
            result = clean_ecg(lead, method="deepfade")
        assert isinstance(result, Lead)
        assert len(result.samples) == len(lead.samples)

    def test_deepfade_with_device(self):
        """Device kwarg is forwarded properly."""
        from ecgdatakit.processing.clean import _DEEPFADE_WEIGHTS
        lead = make_noisy_ecg(duration=10.0)
        if not _DEEPFADE_WEIGHTS.exists():
            pytest.skip("weights file not found")
        with pytest.warns(UserWarning, match="experimental"):
            result = clean_ecg(lead, method="deepfade", device="cpu")
        assert isinstance(result, Lead)


class TestCleanECGOptions:
    def test_powerline_60(self):
        fs = 500
        t = np.arange(0, 10, 1 / fs)
        lead = Lead(label="II", samples=np.sin(2 * np.pi * 60 * t), sampling_rate=fs)
        def hum(result):
            spectrum = np.abs(np.fft.rfft(result.samples[fs:-fs]))
            return spectrum[np.argmin(np.abs(np.fft.rfftfreq(len(t) - 2 * fs, 1 / fs) - 60))]

        assert hum(clean_ecg(lead, powerline=60)) < hum(clean_ecg(lead)) / 100

    def test_unknown_kwarg_rejected(self):
        with pytest.raises(TypeError, match="notch_freq"):
            clean_ecg(make_noisy_ecg(), notch_freq=60)

    def test_deepfade_kwarg_rejected_for_default(self):
        with pytest.raises(TypeError, match="device"):
            clean_ecg(make_noisy_ecg(), device="cpu")

    def test_low_rate_skips_notch_with_warning(self):
        lead = make_noisy_ecg(fs=100)
        with pytest.warns(UserWarning, match="skipped"):
            clean_ecg(lead)

    def test_nan_refused(self):
        lead = make_noisy_ecg()
        lead.samples[100] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            clean_ecg(lead)


class TestCleanECGNeuroKitErrors:
    @pytest.fixture(autouse=True)
    def skip_if_no_nk(self):
        pytest.importorskip("neurokit2")

    def test_backend_error_propagates(self):
        with pytest.raises(ValueError):
            clean_ecg(np.arange(5.0), method="neurokit2", fs=500)

    def test_powerline_forwarded(self):
        import neurokit2 as nk
        lead = make_noisy_ecg()
        expected = nk.ecg_clean(lead.samples, sampling_rate=500, powerline=60)
        np.testing.assert_allclose(clean_ecg(lead, method="neurokit2", powerline=60).samples, expected)


class TestCleanECGRound2:
    def test_notch_warning_points_at_caller(self):
        with pytest.warns(UserWarning, match="skipped") as record:
            clean_ecg(make_noisy_ecg(fs=100))
        assert {w.filename for w in record} == {__file__}

    def test_low_rate_band_clamped_with_warning(self):
        lead = make_noisy_ecg(fs=60)
        with pytest.warns(UserWarning) as record:
            result = clean_ecg(lead)
        messages = [str(w.message) for w in record]
        assert any("upper cutoff lowered from 40.0 Hz to 27 Hz" in m for m in messages)
        assert all(w.filename == __file__ for w in record)
        assert len(result.samples) == len(lead.samples)

    def test_missing_backend_keeps_import_error_and_method(self, monkeypatch):
        import sys
        # None in sys.modules makes the import fail, like a missing peakutils
        monkeypatch.setitem(sys.modules, "biosppy.signals.ecg", None)
        with pytest.raises(ImportError) as info:
            clean_ecg(make_noisy_ecg(), method="combined")
        message = str(info.value)
        assert "method='combined'" in message
        assert "biosppy.signals.ecg" in message
        assert 'pip install "ecgdatakit[cleaning]"' in message
