"""Tests for ecgdatakit.processing.denoise internals and DeepFADE model."""

import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.denoise import _segment, _reassemble


class TestSegment:
    def test_exact_multiple(self):
        signal = np.arange(100, dtype=np.float64)
        segments = _segment(signal, 50)
        assert len(segments) == 2
        np.testing.assert_array_equal(segments[0], np.arange(50))
        np.testing.assert_array_equal(segments[1], np.arange(50, 100))

    def test_with_remainder(self):
        signal = np.arange(75, dtype=np.float64)
        segments = _segment(signal, 50)
        assert len(segments) == 2
        np.testing.assert_array_equal(segments[0], np.arange(50))
        expected = np.zeros(50, dtype=np.float64)
        expected[:25] = np.arange(50, 75)
        np.testing.assert_array_equal(segments[1], expected)

    def test_shorter_than_segment(self):
        signal = np.arange(10, dtype=np.float64)
        segments = _segment(signal, 50)
        assert len(segments) == 1
        assert len(segments[0]) == 50
        np.testing.assert_array_equal(segments[0][:10], np.arange(10))
        np.testing.assert_array_equal(segments[0][10:], 0)

    def test_empty_signal(self):
        signal = np.array([], dtype=np.float64)
        segments = _segment(signal, 50)
        assert len(segments) == 0

    def test_single_segment_exact(self):
        signal = np.arange(50, dtype=np.float64)
        segments = _segment(signal, 50)
        assert len(segments) == 1
        np.testing.assert_array_equal(segments[0], np.arange(50))


class TestReassemble:
    def test_exact_reassembly(self):
        original = np.arange(100, dtype=np.float64)
        segments = np.array([original[:50], original[50:]])
        result = _reassemble(segments, 100, 50)
        np.testing.assert_array_equal(result, original)

    def test_trims_padded_remainder(self):
        original = np.arange(75, dtype=np.float64)
        seg1 = original[:50]
        seg2 = np.zeros(50, dtype=np.float64)
        seg2[:25] = original[50:]
        segments = np.array([seg1, seg2])
        result = _reassemble(segments, 75, 50)
        np.testing.assert_array_equal(result, original)

    def test_single_segment_trim(self):
        original = np.arange(30, dtype=np.float64)
        padded = np.zeros(50, dtype=np.float64)
        padded[:30] = original
        segments = np.array([padded])
        result = _reassemble(segments, 30, 50)
        np.testing.assert_array_equal(result, original)


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
