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
