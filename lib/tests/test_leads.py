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
        assert [l.label for l in result] == ["aVR", "aVL", "aVF"]

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
        labels = [l.label for l in result]
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
