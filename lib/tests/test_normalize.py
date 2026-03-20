import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.normalize import normalize_minmax, normalize_zscore, normalize_amplitude

def make_lead(samples, label="II", fs=500):
    return Lead(label=label, samples=np.array(samples, dtype=np.float64), sampling_rate=fs)

class TestNormalizeMinmax:
    def test_output_range(self):
        lead = make_lead([1, 2, 3, 4, 5])
        result = normalize_minmax(lead)
        assert result.samples.min() == pytest.approx(-1.0)
        assert result.samples.max() == pytest.approx(1.0)

    def test_constant_signal(self):
        lead = make_lead([5, 5, 5, 5])
        result = normalize_minmax(lead)
        np.testing.assert_array_equal(result.samples, np.zeros(4))

    def test_preserves_metadata(self):
        lead = Lead(label="V1", samples=np.array([1.0, 2.0, 3.0]), sampling_rate=250, units="mV")
        result = normalize_minmax(lead)
        assert result.label == "V1"
        assert result.sampling_rate == 250

class TestNormalizeZscore:
    def test_zero_mean_unit_variance(self):
        np.random.seed(42)
        lead = make_lead(np.random.randn(1000) * 5 + 10)
        result = normalize_zscore(lead)
        assert abs(result.samples.mean()) < 0.01
        assert abs(result.samples.std() - 1.0) < 0.01

    def test_constant_signal(self):
        lead = make_lead([3, 3, 3, 3])
        result = normalize_zscore(lead)
        np.testing.assert_array_equal(result.samples, np.zeros(4))

class TestNormalizeAmplitude:
    def test_peak_equals_target(self):
        lead = make_lead([0, 5, -3, 2])
        result = normalize_amplitude(lead, target_mv=1.0)
        assert np.abs(result.samples).max() == pytest.approx(1.0)

    def test_custom_target(self):
        lead = make_lead([0, 10, -5])
        result = normalize_amplitude(lead, target_mv=2.0)
        assert np.abs(result.samples).max() == pytest.approx(2.0)

    def test_constant_zero_signal(self):
        lead = make_lead([0, 0, 0])
        result = normalize_amplitude(lead)
        np.testing.assert_array_equal(result.samples, np.zeros(3))


class TestNormalizeMultipleLeads:
    """Test list[Lead] overloads — per-lead normalization."""

    def _make_leads(self):
        return [
            make_lead([1, 2, 3, 4, 5], label="I"),
            make_lead([10, 20, 30], label="II"),
        ]

    def test_minmax_list(self):
        leads = self._make_leads()
        result = normalize_minmax(leads)
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, Lead)
            assert r.samples.min() == pytest.approx(-1.0)
            assert r.samples.max() == pytest.approx(1.0)
        assert result[0].label == "I"
        assert result[1].label == "II"

    def test_zscore_list(self):
        leads = self._make_leads()
        result = normalize_zscore(leads)
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, Lead)
            assert abs(r.samples.mean()) < 1e-10
            assert abs(r.samples.std() - 1.0) < 0.01

    def test_amplitude_list(self):
        leads = self._make_leads()
        result = normalize_amplitude(leads, target_mv=2.0)
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, Lead)
            assert np.abs(r.samples).max() == pytest.approx(2.0)

    def test_empty_list(self):
        assert normalize_minmax([]) == []
        assert normalize_zscore([]) == []
        assert normalize_amplitude([]) == []


class TestNormalize2DArray:
    """Test 2-D numpy array (n_leads x n_samples) — per-row normalization."""

    def _make_2d(self):
        return np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])

    def test_minmax_2d(self):
        data = self._make_2d()
        result = normalize_minmax(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 5)
        for row in result:
            assert row.min() == pytest.approx(-1.0)
            assert row.max() == pytest.approx(1.0)

    def test_zscore_2d(self):
        data = self._make_2d()
        result = normalize_zscore(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 5)
        for row in result:
            assert abs(row.mean()) < 1e-10
            assert abs(row.std() - 1.0) < 0.01

    def test_amplitude_2d(self):
        data = self._make_2d()
        result = normalize_amplitude(data, target_mv=3.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 5)
        for row in result:
            assert np.abs(row).max() == pytest.approx(3.0)

    def test_1d_array_returns_array(self):
        """1-D numpy in → 1-D numpy out, not a Lead."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result_mm = normalize_minmax(data, fs=500)
        result_zs = normalize_zscore(data, fs=500)
        result_amp = normalize_amplitude(data, fs=500)
        assert isinstance(result_mm, np.ndarray)
        assert isinstance(result_zs, np.ndarray)
        assert isinstance(result_amp, np.ndarray)
        assert result_mm.ndim == 1

    def test_per_row_independence(self):
        """Ensure each row is normalized independently, not across all rows."""
        data = np.array([
            [0.0, 10.0],
            [0.0, 100.0],
        ])
        result = normalize_minmax(data)
        # Both rows should have identical normalized values despite different scales
        np.testing.assert_array_almost_equal(result[0], result[1])


class TestNormalize3DArray:
    """Test 3-D numpy array (n_ecgs x n_leads x n_samples) — per-lead normalization."""

    def _make_3d(self):
        # 2 ECGs, 3 leads each, 5 samples per lead
        return np.array([
            [[1.0, 2.0, 3.0, 4.0, 5.0],
             [10.0, 20.0, 30.0, 40.0, 50.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[5.0, 4.0, 3.0, 2.0, 1.0],
             [50.0, 40.0, 30.0, 20.0, 10.0],
             [100.0, 200.0, 300.0, 400.0, 500.0]],
        ])

    def test_minmax_3d_shape(self):
        data = self._make_3d()
        result = normalize_minmax(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3, 5)

    def test_minmax_3d_per_lead(self):
        data = self._make_3d()
        result = normalize_minmax(data)
        # Each non-constant lead should be in [-1, 1]
        assert result[0, 0].min() == pytest.approx(-1.0)
        assert result[0, 0].max() == pytest.approx(1.0)
        assert result[1, 2].min() == pytest.approx(-1.0)
        assert result[1, 2].max() == pytest.approx(1.0)
        # Constant lead → all zeros
        np.testing.assert_array_equal(result[0, 2], np.zeros(5))

    def test_zscore_3d(self):
        data = self._make_3d()
        result = normalize_zscore(data)
        assert result.shape == (2, 3, 5)
        assert abs(result[0, 0].mean()) < 1e-10
        assert abs(result[1, 2].std() - 1.0) < 0.01

    def test_amplitude_3d(self):
        data = self._make_3d()
        result = normalize_amplitude(data, target_mv=2.0)
        assert result.shape == (2, 3, 5)
        assert np.abs(result[0, 0]).max() == pytest.approx(2.0)
        assert np.abs(result[1, 1]).max() == pytest.approx(2.0)

    def test_3d_independence(self):
        """Each lead normalized independently across ECGs."""
        data = np.array([
            [[0.0, 10.0], [0.0, 100.0]],
            [[0.0, 1.0], [0.0, 1000.0]],
        ])
        result = normalize_minmax(data)
        # All leads should produce the same [-1, 1] result
        np.testing.assert_array_almost_equal(result[0, 0], result[0, 1])
        np.testing.assert_array_almost_equal(result[0, 0], result[1, 0])
        np.testing.assert_array_almost_equal(result[0, 0], result[1, 1])
