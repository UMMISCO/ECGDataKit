import numpy as np
import pytest
from ecgdatakit.processing.hrv import time_domain, frequency_domain, poincare

class TestTimeDomain:
    def test_known_sdnn(self):
        rr = np.array([800.0, 850.0, 780.0, 820.0, 900.0, 770.0, 830.0, 810.0])
        result = time_domain(rr)
        assert result["sdnn"] == pytest.approx(rr.std(ddof=1), abs=0.1)
        assert result["mean_rr"] == pytest.approx(rr.mean(), abs=0.1)

    def test_known_rmssd(self):
        rr = np.array([800.0, 810.0, 790.0, 810.0])
        diffs = np.diff(rr)
        expected_rmssd = np.sqrt(np.mean(diffs ** 2))
        result = time_domain(rr)
        assert result["rmssd"] == pytest.approx(expected_rmssd, abs=0.1)

    def test_pnn50(self):
        # 3 out of 5 diffs exceed 50 ms
        rr = np.array([800.0, 860.0, 800.0, 900.0, 830.0, 900.0])
        diffs = np.abs(np.diff(rr))
        n_over_50 = int(np.sum(diffs > 50))
        result = time_domain(rr)
        assert result["nn50_count"] == n_over_50
        assert result["pnn50"] == pytest.approx(100.0 * n_over_50 / len(diffs), abs=0.1)

    def test_single_interval(self):
        result = time_domain(np.array([800.0]))
        assert result["sdnn"] == 0.0

    def test_hr_mean(self):
        rr = np.array([1000.0, 1000.0, 1000.0])  # 60 bpm
        result = time_domain(rr)
        assert result["hr_mean"] == pytest.approx(60.0, abs=0.1)

class TestFrequencyDomain:
    def test_returns_expected_keys(self):
        rr = np.random.normal(800, 50, 300)
        result = frequency_domain(rr)
        assert "vlf_power" in result
        assert "lf_power" in result
        assert "hf_power" in result
        assert "lf_hf_ratio" in result
        assert "total_power" in result

    def test_short_input(self):
        result = frequency_domain(np.array([800.0, 810.0]))
        assert result["total_power"] == 0.0

class TestPoincare:
    def test_sd1_sd2(self):
        rr = np.array([800.0, 810.0, 790.0, 810.0, 800.0, 820.0, 780.0, 810.0])
        result = poincare(rr)
        assert result["sd1"] > 0
        assert result["sd2"] > 0
        assert "sd1_sd2_ratio" in result

    def test_single_interval(self):
        result = poincare(np.array([800.0]))
        assert result["sd1"] == 0.0
