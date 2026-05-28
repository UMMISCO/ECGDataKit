import numpy as np
import pytest
from ecgdatakit.models import ECGRecord, Lead
from ecgdatakit.processing.normalize import normalize_minmax, normalize_zscore, normalize_amplitude


def make_lead(samples, label="II", fs=500):
    return Lead(label=label, samples=np.array(samples, dtype=np.float64), sampling_rate=fs)


def make_record(lead_data, labels=None):
    """Build an ECGRecord from a list of (label, samples) or raw sample lists."""
    leads = []
    for i, ld in enumerate(lead_data):
        if isinstance(ld, tuple):
            lbl, samp = ld
        else:
            lbl = labels[i] if labels else f"L{i}"
            samp = ld
        leads.append(make_lead(samp, label=lbl))
    return ECGRecord(leads=leads)


# -----------------------------------------------------------------------
# Single Lead
# -----------------------------------------------------------------------

class TestNormalizeMinmaxLead:
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
        assert result.units == "mV"


class TestNormalizeZscoreLead:
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


class TestNormalizeAmplitudeLead:
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


# -----------------------------------------------------------------------
# list[Lead]
# -----------------------------------------------------------------------

class TestNormalizeLeadList:
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

    def test_per_lead_independence(self):
        """Each lead is normalized independently regardless of scale."""
        leads = [
            make_lead([0, 10], label="I"),
            make_lead([0, 100], label="II"),
        ]
        result = normalize_minmax(leads)
        np.testing.assert_array_almost_equal(result[0].samples, result[1].samples)


# -----------------------------------------------------------------------
# ECGRecord
# -----------------------------------------------------------------------

class TestNormalizeECGRecord:
    def _make_record(self):
        return ECGRecord(
            leads=[
                make_lead([1, 2, 3, 4, 5], label="I"),
                make_lead([10, 20, 30, 40, 50], label="II"),
            ],
            median_beats=[
                make_lead([2, 4, 6], label="I"),
            ],
        )

    def test_minmax_record(self):
        rec = self._make_record()
        result = normalize_minmax(rec)
        assert isinstance(result, ECGRecord)
        assert len(result.leads) == 2
        for ld in result.leads:
            assert ld.samples.min() == pytest.approx(-1.0)
            assert ld.samples.max() == pytest.approx(1.0)
        assert result.leads[0].label == "I"
        assert result.leads[1].label == "II"

    def test_minmax_normalizes_median_beats(self):
        rec = self._make_record()
        result = normalize_minmax(rec)
        assert len(result.median_beats) == 1
        assert result.median_beats[0].samples.min() == pytest.approx(-1.0)
        assert result.median_beats[0].samples.max() == pytest.approx(1.0)

    def test_zscore_record(self):
        rec = self._make_record()
        result = normalize_zscore(rec)
        assert isinstance(result, ECGRecord)
        for ld in result.leads:
            assert abs(ld.samples.mean()) < 1e-10
            assert abs(ld.samples.std() - 1.0) < 0.01

    def test_amplitude_record(self):
        rec = self._make_record()
        result = normalize_amplitude(rec, target_mv=3.0)
        assert isinstance(result, ECGRecord)
        for ld in result.leads:
            assert np.abs(ld.samples).max() == pytest.approx(3.0)

    def test_preserves_record_metadata(self):
        rec = self._make_record()
        rec = ECGRecord(
            leads=rec.leads,
            source_format="test",
            annotations={"key": "value"},
        )
        result = normalize_minmax(rec)
        assert result.source_format == "test"
        assert result.annotations == {"key": "value"}

    def test_per_lead_independence(self):
        """Each lead in the record is normalized independently."""
        rec = ECGRecord(leads=[
            make_lead([0, 10], label="I"),
            make_lead([0, 100], label="II"),
        ])
        result = normalize_minmax(rec)
        np.testing.assert_array_almost_equal(
            result.leads[0].samples, result.leads[1].samples
        )

    def test_empty_leads(self):
        rec = ECGRecord(leads=[])
        result = normalize_minmax(rec)
        assert isinstance(result, ECGRecord)
        assert result.leads == []


# -----------------------------------------------------------------------
# list[ECGRecord]
# -----------------------------------------------------------------------

class TestNormalizeECGRecordList:
    def _make_records(self):
        return [
            ECGRecord(leads=[
                make_lead([1, 2, 3, 4, 5], label="I"),
                make_lead([10, 20, 30], label="II"),
            ]),
            ECGRecord(leads=[
                make_lead([5, 4, 3, 2, 1], label="I"),
                make_lead([50, 40, 30, 20, 10], label="II"),
            ]),
        ]

    def test_minmax_record_list(self):
        records = self._make_records()
        result = normalize_minmax(records)
        assert isinstance(result, list)
        assert len(result) == 2
        for rec in result:
            assert isinstance(rec, ECGRecord)
            for ld in rec.leads:
                assert ld.samples.min() == pytest.approx(-1.0)
                assert ld.samples.max() == pytest.approx(1.0)

    def test_zscore_record_list(self):
        records = self._make_records()
        result = normalize_zscore(records)
        assert len(result) == 2
        for rec in result:
            for ld in rec.leads:
                assert abs(ld.samples.mean()) < 1e-10
                assert abs(ld.samples.std() - 1.0) < 0.01

    def test_amplitude_record_list(self):
        records = self._make_records()
        result = normalize_amplitude(records, target_mv=2.0)
        assert len(result) == 2
        for rec in result:
            for ld in rec.leads:
                assert np.abs(ld.samples).max() == pytest.approx(2.0)

    def test_empty_record_list(self):
        assert normalize_minmax([]) == []

    def test_per_ecg_independence(self):
        """Each ECG record is normalized independently."""
        records = [
            ECGRecord(leads=[make_lead([0, 10], label="I")]),
            ECGRecord(leads=[make_lead([0, 1000], label="I")]),
        ]
        result = normalize_minmax(records)
        np.testing.assert_array_almost_equal(
            result[0].leads[0].samples,
            result[1].leads[0].samples,
        )


# -----------------------------------------------------------------------
# 3-D numpy array (n_ecgs, n_leads, n_samples)
# -----------------------------------------------------------------------

class TestNormalize3DArray:
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

    def test_per_lead_per_ecg_independence(self):
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


# -----------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------

class TestNormalizeErrors:
    def test_rejects_1d_numpy(self):
        with pytest.raises(ValueError, match="must be 3-D"):
            normalize_minmax(np.array([1.0, 2.0, 3.0]))

    def test_rejects_2d_numpy(self):
        with pytest.raises(ValueError, match="must be 3-D"):
            normalize_zscore(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="Expected Lead"):
            normalize_zscore("not valid")