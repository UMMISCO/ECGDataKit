import numpy as np
import pytest
from ecgdatakit.models import Lead
from ecgdatakit.processing.leads import derive_lead_iii, derive_augmented, derive_standard_12, find_lead

def make_lead(label, values=None, fs=500):
    if values is None:
        np.random.seed(hash(label) % 2**31)
        values = np.random.randn(1000)
    return Lead(label=label, samples=np.array(values, dtype=np.float64), sampling_rate=fs)

class TestDeriveLeadIII:
    def test_iii_equals_ii_minus_i(self):
        i_vals = np.array([1.0, 2.0, 3.0, 4.0])
        ii_vals = np.array([5.0, 6.0, 7.0, 8.0])
        lead_i = make_lead("I", i_vals)
        lead_ii = make_lead("II", ii_vals)
        iii = derive_lead_iii(lead_i, lead_ii)
        np.testing.assert_array_almost_equal(iii.samples, ii_vals - i_vals)
        assert iii.label == "III"

    def test_sampling_rate_mismatch_raises(self):
        a = Lead(label="I", samples=np.zeros(10, dtype=np.float64), sampling_rate=500)
        b = Lead(label="II", samples=np.zeros(10, dtype=np.float64), sampling_rate=250)
        with pytest.raises(ValueError, match="Sample rates"):
            derive_lead_iii(a, b)

    def test_length_mismatch_raises(self):
        a = Lead(label="I", samples=np.zeros(10, dtype=np.float64), sampling_rate=500)
        b = Lead(label="II", samples=np.zeros(20, dtype=np.float64), sampling_rate=500)
        with pytest.raises(ValueError, match="Sample counts"):
            derive_lead_iii(a, b)

class TestDeriveAugmented:
    def test_returns_three_leads(self):
        lead_i = make_lead("I", np.ones(100))
        lead_ii = make_lead("II", np.ones(100) * 2)
        result = derive_augmented(lead_i, lead_ii)
        assert len(result) == 3
        assert [ld.label for ld in result] == ["aVR", "aVL", "aVF"]

    def test_avr_formula(self):
        i_vals = np.array([2.0, 4.0])
        ii_vals = np.array([6.0, 8.0])
        lead_i = make_lead("I", i_vals)
        lead_ii = make_lead("II", ii_vals)
        avr, avl, avf = derive_augmented(lead_i, lead_ii)
        # aVR = -(I + II) / 2
        np.testing.assert_array_almost_equal(avr.samples, -(i_vals + ii_vals) / 2)
        # aVL = I - II/2
        np.testing.assert_array_almost_equal(avl.samples, i_vals - ii_vals / 2)
        # aVF = II - I/2
        np.testing.assert_array_almost_equal(avf.samples, ii_vals - i_vals / 2)

class TestDeriveStandard12:
    def test_returns_12_leads(self):
        leads = {name: make_lead(name, np.ones(100)) for name in ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]}
        result = derive_standard_12(leads["I"], leads["II"], leads["V1"], leads["V2"], leads["V3"], leads["V4"], leads["V5"], leads["V6"])
        assert len(result) == 12
        labels = [ld.label for ld in result]
        assert labels == ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

class TestFindLead:
    def test_finds_case_insensitive(self):
        leads = [make_lead("I"), make_lead("II"), make_lead("aVL")]
        assert find_lead(leads, "avl").label == "aVL"
        assert find_lead(leads, "AVL").label == "aVL"
        assert find_lead(leads, "aVL").label == "aVL"

    def test_returns_none_if_not_found(self):
        leads = [make_lead("I"), make_lead("II")]
        assert find_lead(leads, "V6") is None


class TestDeriveScaling:
    def _raw(self, label, counts, res, offset=0.0, unit="mV"):
        return Lead(label=label, samples=np.array(counts, dtype=np.float64), sampling_rate=500,
                    resolution=res, resolution_unit=unit, offset=offset, is_raw=True)

    def test_raw_leads_with_different_gains_use_physical_values(self):
        # WFDB-like leads with per-lead gain and baseline
        lead_i = self._raw("I", [100.0, 200.0], 0.002, offset=0.01)
        lead_ii = self._raw("II", [100.0, 300.0], 0.005, offset=-0.02)
        phys_i = lead_i.to_physical().samples
        phys_ii = lead_ii.to_physical().samples
        iii = derive_lead_iii(lead_i, lead_ii)
        assert not iii.is_raw and iii.units == "mV" and iii.offset == 0.0
        np.testing.assert_allclose(iii.samples, phys_ii - phys_i)
        avr, avl, avf = derive_augmented(lead_i, lead_ii)
        np.testing.assert_allclose(avr.samples, -(phys_i + phys_ii) / 2)
        np.testing.assert_allclose(avl.samples, phys_i - phys_ii / 2)
        np.testing.assert_allclose(avf.samples, phys_ii - phys_i / 2)

    def test_raw_leads_same_scale_stay_raw_with_offset_folded(self):
        lead_i = self._raw("I", [10.0, 20.0], 0.005, offset=0.5)
        lead_ii = self._raw("II", [30.0, 50.0], 0.005, offset=0.5)
        iii = derive_lead_iii(lead_i, lead_ii)
        avr = derive_augmented(lead_i, lead_ii)[0]
        assert iii.is_raw and iii.resolution == 0.005 and iii.offset == 0.0
        expected_iii = lead_ii.to_physical().samples - lead_i.to_physical().samples
        np.testing.assert_allclose(iii.to_physical().samples, expected_iii)
        expected_avr = -(lead_i.to_physical().samples + lead_ii.to_physical().samples) / 2
        np.testing.assert_allclose(avr.to_physical().samples, expected_avr)

    def test_mixed_raw_and_physical(self):
        lead_i = self._raw("I", [1000.0, 2000.0], 1.0, unit="uV")
        lead_ii = Lead(label="II", samples=np.array([3.0, 5.0]), sampling_rate=500,
                       units="mV", resolution_unit="mV", is_raw=False)
        iii = derive_lead_iii(lead_i, lead_ii)
        assert iii.units == "uV"
        np.testing.assert_allclose(iii.samples, [2000.0, 3000.0])

    def test_raw_without_unit_and_different_scale_raises(self):
        lead_i = Lead(label="I", samples=np.zeros(4), sampling_rate=500, resolution=2.0)
        lead_ii = Lead(label="II", samples=np.zeros(4), sampling_rate=500, resolution=3.0)
        with pytest.raises(ValueError, match="voltage scale"):
            derive_lead_iii(lead_i, lead_ii)

    def test_derived_leads_drop_file_metadata(self):
        lead_i = make_lead("I")
        lead_i.annotations = {"qrs_onset": "120"}
        lead_i.quality = 3
        lead_i.adc_resolution = 4.88
        for lead in [derive_lead_iii(lead_i, make_lead("II")), *derive_augmented(lead_i, make_lead("II"))]:
            assert lead.annotations == {} and lead.quality is None
            assert lead.adc_resolution == 0.0

    def test_int16_samples_do_not_overflow(self):
        lead_i = Lead(label="I", samples=np.array([30000, 30000], dtype=np.int16), sampling_rate=500)
        lead_ii = Lead(label="II", samples=np.array([30000, -30000], dtype=np.int16), sampling_rate=500)
        np.testing.assert_array_equal(derive_augmented(lead_i, lead_ii)[0].samples, [-30000.0, 0.0])
        np.testing.assert_array_equal(derive_lead_iii(lead_i, lead_ii).samples, [0.0, -60000.0])

    def test_standard_12_converts_precordials_with_limbs(self):
        lead_i = self._raw("I", [100.0, 200.0], 0.002)
        lead_ii = self._raw("II", [100.0, 300.0], 0.005)
        vs = [self._raw(f"V{k}", [1000.0, 1000.0], 1.0, unit="uV") for k in range(1, 7)]
        leads = derive_standard_12(lead_i, lead_ii, *vs)
        assert all(not ld.is_raw and ld.units == "mV" for ld in leads)
        np.testing.assert_allclose(leads[6].samples, [1.0, 1.0])

    def test_standard_12_checks_precordial_length(self):
        leads = [make_lead(lbl) for lbl in ("I", "II", "V1", "V2", "V3", "V4", "V5")]
        with pytest.raises(ValueError, match="Sample counts"):
            derive_standard_12(*leads, make_lead("V6", np.zeros(10)))
