"""Tests for ecgdatakit.processing.denoise internals and DeepFADE model."""

import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.denoise import _overlap_add, _pad, _rescale

_WAVES = [(-0.20, 0.15, 0.025), (-0.03, -0.10, 0.010), (0.0, 1.0, 0.010),
          (0.03, -0.25, 0.010), (0.25, 0.30, 0.050)]


def synth_ecg(fs=500, duration=20.0, bpm=70):
    t = np.arange(int(duration * fs)) / fs
    x = np.zeros_like(t)
    pos = 0.6
    while pos < duration - 0.6:
        for off, amp, w in _WAVES:
            x += amp * np.exp(-0.5 * ((t - pos - off) / w) ** 2)
        pos += 60.0 / bpm
    return x


def noisy_ecg(fs=500, duration=20.0):
    rng = np.random.default_rng(0)
    clean = synth_ecg(fs, duration)
    t = np.arange(len(clean)) / fs
    noise = (0.5 * np.sin(2 * np.pi * 0.25 * t) + 0.05 * rng.standard_normal(len(t))
             + 0.05 * np.sin(2 * np.pi * 50 * t))
    return clean, clean + noise


class TestPad:
    @pytest.mark.parametrize("n", [500, 2500, 5000, 7499, 12345])
    def test_whole_windows_and_full_coverage(self, n):
        x = np.arange(n, dtype=np.float64)
        padded, left = _pad(x, 5000, 2500)
        assert left == 2500
        assert (len(padded) - 5000) % 2500 == 0
        np.testing.assert_array_equal(padded[left : left + n], x)
        assert len(padded) - left - n >= 2500


class TestOverlapAdd:
    def test_constant_segments_reconstruct_exactly(self):
        length = 5000 + 3 * 2500
        starts = list(range(0, length - 5000 + 1, 2500))
        segs = np.full((len(starts), 5000), 2.0)
        out = _overlap_add(segs, starts, length)
        np.testing.assert_allclose(out[2500:-2500], 2.0)

    def test_matching_segments_reconstruct_signal(self):
        x = np.sin(np.arange(12500) / 50.0)
        starts = list(range(0, 12500 - 5000 + 1, 2500))
        segs = np.stack([x[s : s + 5000] for s in starts])
        np.testing.assert_allclose(_overlap_add(segs, starts, 12500)[2500:-2500], x[2500:-2500])


class TestRescale:
    def test_recovers_gain_and_zero_median(self):
        ref = np.sin(np.arange(5000) / 30.0)
        out = _rescale(4.0 * ref + 3.0, ref)
        np.testing.assert_allclose(out, ref - np.median(ref), atol=1e-12)

    def test_flat_output_gives_zeros(self):
        assert np.all(_rescale(np.ones(5000), np.arange(5000.0)) == 0)


class TestRequireTorch:
    def test_import_error_message(self):
        """Helpful error raised when torch is missing."""
        try:
            import torch  # noqa: F401
            pytest.skip("torch is installed")
        except ImportError:
            from ecgdatakit.processing.denoise import _require_torch
            with pytest.raises(ImportError, match="torch is required"):
                _require_torch()


class TestDeepFADEModel:
    """Tests for the DeepFADE model architecture."""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_model_instantiation(self):
        from ecgdatakit.processing.nn.deepfade import DeepFADE
        model = DeepFADE(**DeepFADE.DEFAULT_ARGS)
        assert model is not None

    def test_model_forward_shape(self):
        import torch
        from ecgdatakit.processing.nn.deepfade import DeepFADE

        model = DeepFADE(**DeepFADE.DEFAULT_ARGS)
        model.eval()
        x = torch.randn(2, 1, 5000)
        with torch.no_grad():
            clean, baseline = model(x)
        assert clean.shape == (2, 1, 5000)
        assert baseline.shape == (2, 1, 5000)

    def test_model_output_types(self):
        import torch
        from ecgdatakit.processing.nn.deepfade import DeepFADE

        model = DeepFADE(**DeepFADE.DEFAULT_ARGS)
        model.eval()
        x = torch.randn(1, 1, 5000)
        with torch.no_grad():
            clean, baseline = model(x)
        assert clean.dtype == torch.float32
        assert baseline.dtype == torch.float32

    def test_encoder_output_channels(self):
        from ecgdatakit.processing.nn.deepfade import DenseEncoder

        enc = DenseEncoder(
            input_channels=1,
            pool_steps=[2, 2, 2, 5],
            layers=8,
            compression=1,
            bottleneck=False,
            activation={"name": "elu", "args": {"alpha": 0.1}},
            dropout_rate=0.2,
            pool_type="convolution",
        )
        assert enc.get_output_channels() == 8

    def test_dense_trunk_builds(self):
        from ecgdatakit.processing.nn.dense_net import DenseTrunk

        trunk = DenseTrunk(
            input_channels=1,
            blocks=3,
            layers=4,
            growth_rate=12,
        )
        assert trunk.get_output_channels() > 0


class TestDenoiseDeepfade:
    """End-to-end tests with the bundled weights."""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch")

    @staticmethod
    def run(lead, **kwargs):
        from ecgdatakit.processing.clean import _DEEPFADE_WEIGHTS
        from ecgdatakit.processing.denoise import denoise_deepfade

        with pytest.warns(UserWarning, match="experimental"):
            return denoise_deepfade(lead, _DEEPFADE_WEIGHTS, **kwargs)

    def test_bundled_weights_load_strictly(self):
        import torch
        from ecgdatakit.processing.clean import _DEEPFADE_WEIGHTS
        from ecgdatakit.processing.denoise import _remap_state_dict_key
        from ecgdatakit.processing.nn.deepfade import DeepFADE

        assert _DEEPFADE_WEIGHTS.is_file()
        state = torch.load(_DEEPFADE_WEIGHTS, map_location="cpu", weights_only=True)
        model = DeepFADE(**DeepFADE.DEFAULT_ARGS)
        result = model.load_state_dict({_remap_state_dict_key(k): v for k, v in state.items()})
        assert not result.missing_keys and not result.unexpected_keys

    def test_amplitude_preserved_and_noise_reduced(self):
        clean, noisy = noisy_ecg()
        out = self.run(noisy, fs=500).samples
        ref = clean - np.median(clean)
        assert 0.8 < np.ptp(out) / np.ptp(clean) < 1.25
        rmse_out = np.sqrt(np.mean((out - ref) ** 2))
        rmse_in = np.sqrt(np.mean((noisy - np.median(noisy) - ref) ** 2))
        assert rmse_out < 0.5 * rmse_in

    def test_scale_and_offset_equivariant(self):
        _, noisy = noisy_ecg(duration=10)
        base = self.run(noisy, fs=500).samples
        scaled = self.run(1000.0 * noisy + 5.0, fs=500).samples / 1000.0
        np.testing.assert_allclose(scaled, base, atol=1e-9 * np.ptp(base))

    @pytest.mark.parametrize("fs, n", [(500, 500), (500, 7499), (360, 9001), (250, 6250)])
    def test_exact_length_and_metadata(self, fs, n):
        _, noisy = noisy_ecg(fs=fs, duration=n / fs + 1)
        lead = Lead(label="II", samples=noisy[:n], sampling_rate=fs, units="mV",
                    resolution_unit="mV", is_raw=False)
        out = self.run(lead)
        assert len(out.samples) == n
        assert out.sampling_rate == fs and out.units == "mV" and out.label == "II"

    def test_deterministic(self):
        _, noisy = noisy_ecg(duration=10)
        np.testing.assert_array_equal(self.run(noisy, fs=500).samples,
                                      self.run(noisy, fs=500).samples)

    def test_raw_counts_refused(self):
        _, noisy = noisy_ecg(duration=10)
        raw = Lead(label="II", samples=np.round(noisy / 0.005), sampling_rate=500,
                   resolution=0.005, resolution_unit="mV", is_raw=True)
        with pytest.raises(ValueError, match="raw ADC counts"):
            self.run(raw)

    @pytest.mark.parametrize("bad, match", [
        (np.array([]), "too short"),
        (np.zeros(100), "too short"),
        (np.r_[np.zeros(1000), np.nan], "NaN"),
    ])
    def test_invalid_input(self, bad, match):
        with pytest.raises(ValueError, match=match):
            self.run(bad, fs=500)

    def test_warning_points_at_caller(self):
        from ecgdatakit.processing import clean_ecg

        _, noisy = noisy_ecg(duration=10)
        with pytest.warns(UserWarning, match="experimental") as rec:
            clean_ecg(noisy, method="deepfade", fs=500)
        assert rec[0].filename == __file__


class TestDeepfadeOffsetAndTorch:
    @pytest.mark.parametrize("fs", [360, 180])
    def test_offset_invariant_when_resampling(self, fs):
        pytest.importorskip("torch")
        import warnings

        from ecgdatakit.processing.clean import _DEEPFADE_WEIGHTS
        from ecgdatakit.processing.denoise import denoise_deepfade

        _, noisy = noisy_ecg(fs=fs, duration=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            base = denoise_deepfade(noisy, _DEEPFADE_WEIGHTS, fs=fs).samples
            shifted = denoise_deepfade(noisy + 1000.0, _DEEPFADE_WEIGHTS, fs=fs).samples
        # float32 inference: only rounding-level differences remain
        np.testing.assert_allclose(shifted, base, atol=1e-6 * np.ptp(base))

    def test_no_experimental_warning_without_torch(self, monkeypatch, recwarn):
        import ecgdatakit.processing.denoise as dn

        def missing():
            raise ImportError("torch is required for DeepFADE denoising.")

        monkeypatch.setattr(dn, "_require_torch", missing)
        with pytest.raises(ImportError, match="torch is required"):
            dn.denoise_deepfade(np.zeros(1000), "unused.pt", fs=500)
        assert not [w for w in recwarn if "experimental" in str(w.message)]
