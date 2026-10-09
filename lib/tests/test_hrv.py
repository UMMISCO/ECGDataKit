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
        assert result["mean_rr"] == 800.0
        assert result["hr_mean"] == pytest.approx(75.0)
        for key in ("sdnn", "rmssd", "sdsd", "pnn50", "pnn20", "hr_std"):
            assert np.isnan(result[key]), key

    def test_empty(self):
        result = time_domain(np.array([]))
        assert np.isnan(result["mean_rr"]) and np.isnan(result["sdnn"])
        assert result["nn50_count"] == 0

    def test_two_intervals(self):
        result = time_domain([800.0, 900.0])
        assert result["rmssd"] == pytest.approx(100.0)
        assert result["pnn50"] == 100.0
        assert np.isnan(result["sdsd"])

    def test_pnn50_strictly_greater(self):
        result = time_domain(np.array([800.0, 850.0, 800.0]))
        assert result["nn50_count"] == 0

    @pytest.mark.parametrize("bad", [[800.0, np.nan, 810.0], [800.0, 0.0, 810.0],
                                     [800.0, -5.0], [[800.0, 810.0], [790.0, 800.0]]])
    def test_invalid_input_raises(self, bad):
        with pytest.raises(ValueError):
            time_domain(np.array(bad))

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
        assert all(np.isnan(v) for v in result.values())

    @staticmethod
    def sinusoidal_rr(freq, amp=50.0, duration=300.0, mean=800.0):
        rr, t = [], 0.0
        while t < duration:
            r = mean + amp * np.sin(2 * np.pi * freq * t)
            rr.append(r)
            t += r / 1000.0
        return np.array(rr)

    def test_hf_sinusoid_power_in_ms2(self):
        # a 50 ms sinusoid has variance 50**2 / 2 = 1250 ms^2
        result = frequency_domain(self.sinusoidal_rr(0.25))
        assert result["hf_power"] == pytest.approx(1250.0, rel=0.05)
        assert result["lf_power"] < 0.02 * result["hf_power"]
        assert result["total_power"] == pytest.approx(1250.0, rel=0.05)

    def test_lf_sinusoid(self):
        result = frequency_domain(self.sinusoidal_rr(0.10))
        assert result["lf_power"] == pytest.approx(1250.0, rel=0.05)
        assert result["hf_power"] < 0.02 * result["lf_power"]
        assert result["lf_hf_ratio"] > 50

    def test_no_warning_for_5_minutes(self, recwarn):
        frequency_domain(self.sinusoidal_rr(0.25))
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_warns_below_2_minutes(self):
        with pytest.warns(UserWarning, match="LF power needs"):
            frequency_domain(self.sinusoidal_rr(0.25, duration=90))

    def test_warns_below_1_minute(self):
        with pytest.warns(UserWarning, match="LF and HF"):
            frequency_domain(self.sinusoidal_rr(0.25, duration=10))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            frequency_domain(self.sinusoidal_rr(0.25), method="lomb")

    @pytest.mark.parametrize("interp_fs", [float("nan"), float("inf"), 0.0, -4.0, 0.5])
    def test_invalid_interp_fs_raises(self, interp_fs):
        with pytest.raises(ValueError, match="interp_fs"):
            frequency_domain(self.sinusoidal_rr(0.25), interp_fs=interp_fs)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            frequency_domain(np.array([800.0, np.nan, 790.0, 850.0, 800.0]))

class TestPoincare:
    def test_sd1_sd2(self):
        rr = np.array([800.0, 810.0, 790.0, 810.0, 800.0, 820.0, 780.0, 810.0])
        result = poincare(rr)
        assert result["sd1"] > 0
        assert result["sd2"] > 0
        assert "sd1_sd2_ratio" in result

    def test_known_values(self):
        result = poincare(np.array([800.0, 810.0, 790.0]))
        assert result["sd1"] == pytest.approx(15.0)
        assert result["sd2"] == pytest.approx(5.0)
        assert result["sd1_sd2_ratio"] == pytest.approx(3.0)

    @pytest.mark.parametrize("rr", [[800.0], [800.0, 810.0]])
    def test_too_few_intervals_nan(self, rr):
        result = poincare(np.array(rr))
        assert all(np.isnan(v) for v in result.values())
