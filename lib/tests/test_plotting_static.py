"""Tests for ecgdatakit.plotting.static (matplotlib plots)."""

from __future__ import annotations

import numpy as np
import pytest

from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    GlobalMeasurements,
    Interpretation,
    Lead,
    PatientInfo,
    RecordingInfo,
)

matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt  # noqa: E402

from ecgdatakit.plotting.static import (  # noqa: E402
    plot_12lead,
    plot_average_beat,
    plot_beats,
    plot_hrv_summary,
    plot_lead,
    plot_leads,
    plot_peaks,
    plot_poincare,
    plot_quality,
    plot_report,
    plot_rr_tachogram,
    plot_spectrogram,
    plot_spectrum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_lead(label: str = "II", fs: int = 500, duration: float = 5.0) -> Lead:
    """Create a synthetic lead with a 10 Hz sine + noise."""
    n = int(fs * duration)
    t = np.arange(n, dtype=np.float64) / fs
    samples = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.default_rng(42).standard_normal(n)
    return Lead(label=label, samples=samples.astype(np.float64), sampling_rate=fs, units="mV")


def _make_ecg_lead(label: str = "II", fs: int = 500, duration: float = 5.0) -> Lead:
    """Create a synthetic lead with QRS-like peaks for R-peak detection."""
    n = int(fs * duration)
    signal = np.zeros(n, dtype=np.float64)
    # Add QRS complexes every ~0.8 seconds (75 bpm)
    rr_samples = int(0.8 * fs)
    for pos in range(rr_samples, n, rr_samples):
        lo = max(0, pos - int(0.02 * fs))
        hi = min(n, pos + int(0.02 * fs))
        for j in range(lo, hi):
            signal[j] = 1.5 * np.exp(-((j - pos) / (0.008 * fs)) ** 2)
    signal += 0.05 * np.random.default_rng(42).standard_normal(n)
    return Lead(label=label, samples=signal, sampling_rate=fs, units="mV")


def _make_12leads(fs: int = 500, duration: float = 5.0) -> list[Lead]:
    """Create all 12 leads."""
    labels = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    return [_make_ecg_lead(lbl, fs, duration) for lbl in labels]


def _make_record(fs: int = 500, duration: float = 5.0) -> ECGRecord:
    """Create a full ECGRecord with metadata."""
    rec = RecordingInfo()
    rec.acquisition.signal.sampling_rate = fs
    rec.device = DeviceInfo(manufacturer="TestCo", model="ECG-1000")
    rec.acquisition.filters = FilterSettings(highpass=0.05, lowpass=150.0)
    return ECGRecord(
        patient=PatientInfo(patient_id="P001", first_name="John", last_name="Doe", age=55, sex="M"),
        recording=rec,
        leads=_make_12leads(fs, duration),
        interpretation=Interpretation(statements=[("Normal sinus rhythm", "")], severity="NORMAL"),
        measurements=GlobalMeasurements(heart_rate=75, pr_interval=160, qrs_duration=90, qt_interval=380, qtc_bazett=410, qrs_axis=60),
        source_format="test",
    )


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlotLead:
    def test_returns_figure(self):
        fig = plot_lead(_make_lead())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_peaks(self):
        peaks = np.array([100, 500, 900], dtype=np.intp)
        fig = plot_lead(_make_lead(), peaks=peaks)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_ax(self):
        _, ax = plt.subplots()
        fig = plot_lead(_make_lead(), ax=ax)
        assert fig is ax.get_figure()

    def test_no_grid(self):
        fig = plot_lead(_make_lead(), show_grid=False)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotLeads:
    def test_returns_figure(self):
        leads = [_make_lead("I"), _make_lead("II"), _make_lead("III")]
        fig = plot_leads(leads)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_accepts_ecgrecord(self):
        record = _make_record()
        fig = plot_leads(record)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_leads(self):
        fig = plot_leads([])
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_peaks_dict(self):
        leads = [_make_lead("I"), _make_lead("II")]
        peaks_dict = {"I": np.array([100, 500], dtype=np.intp)}
        fig = plot_leads(leads, peaks_dict=peaks_dict)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlot12Lead:
    def test_returns_figure(self):
        leads = _make_12leads()
        fig = plot_12lead(leads)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_record(self):
        record = _make_record()
        fig = plot_12lead(record)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_explicit_record(self):
        leads = _make_12leads()
        record = _make_record()
        fig = plot_12lead(leads, record=record)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotPeaks:
    def test_returns_figure(self):
        lead = _make_ecg_lead()
        fig = plot_peaks(lead)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_explicit_peaks(self):
        lead = _make_lead()
        peaks = np.array([100, 500, 900], dtype=np.intp)
        fig = plot_peaks(lead, peaks=peaks)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotBeats:
    def test_overlay(self):
        lead = _make_ecg_lead()
        fig = plot_beats(lead, overlay=True)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_waterfall(self):
        lead = _make_ecg_lead()
        fig = plot_beats(lead, overlay=False)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotAverageBeat:
    def test_returns_figure(self):
        lead = _make_ecg_lead()
        fig = plot_average_beat(lead)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotSpectrum:
    def test_welch(self):
        fig = plot_spectrum(_make_lead(), method="welch")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_fft(self):
        fig = plot_spectrum(_make_lead(), method="fft")
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotSpectrogram:
    def test_returns_figure(self):
        fig = plot_spectrogram(_make_lead())
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotRRTachogram:
    def test_returns_figure(self):
        rr = np.array([800, 810, 790, 820, 780, 800, 815], dtype=np.float64)
        fig = plot_rr_tachogram(rr)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotPoincare:
    def test_returns_figure(self):
        rr = np.array([800, 810, 790, 820, 780, 800, 815], dtype=np.float64)
        fig = plot_poincare(rr)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_short_rr(self):
        rr = np.array([800], dtype=np.float64)
        fig = plot_poincare(rr)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotHRVSummary:
    def test_returns_figure(self):
        rr = np.random.default_rng(42).normal(800, 30, 50).astype(np.float64)
        with pytest.warns(UserWarning, match="unreliable") as rec:
            fig = plot_hrv_summary(rr)
        assert rec[0].filename == __file__
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotQuality:
    def test_returns_figure(self):
        leads = [_make_lead("I"), _make_lead("II"), _make_lead("V1")]
        fig = plot_quality(leads)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_accepts_ecgrecord(self):
        fig = plot_quality(_make_record())
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotReport:
    def test_returns_figure(self):
        record = _make_record()
        fig = plot_report(record)
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# Axes, units and edge cases
# ---------------------------------------------------------------------------


def _raw_lead(label: str = "II", n: int = 2500, fs: int = 500) -> Lead:
    """Raw counts as a parser returns them with auto_scale=False."""
    return Lead(label=label, samples=np.arange(n, dtype=np.float64) % 200,
                sampling_rate=fs, resolution=5.0, resolution_unit="uV")


class TestTimeAxis:
    def test_plot_lead_time_axis_ends_at_last_sample(self):
        lead = _make_lead(fs=250, duration=4.0)
        ax = plot_lead(lead, show=False).axes[0]
        x = ax.get_lines()[0].get_xdata()
        assert x[1] == pytest.approx(1 / 250)
        assert x[-1] == pytest.approx(999 / 250)
        assert ax.get_xlim() == pytest.approx((0.0, 999 / 250))

    def test_samples_axis(self):
        ax = plot_lead(_make_lead(fs=250, duration=1.0), show=False, x_axis="samples").axes[0]
        assert ax.get_lines()[0].get_xdata()[[0, -1]].tolist() == [1, 250]
        assert ax.get_xlabel() == "Sample"

    def test_beats_axis_is_relative_to_r_peak(self):
        lead = _make_ecg_lead(fs=500, duration=5.0)
        peaks = np.array([400, 800, 1200, 1600])
        ax = plot_beats(lead, peaks=peaks, show=False).axes[0]
        x = ax.get_lines()[0].get_xdata()
        assert x[0] == pytest.approx(-200.0)
        assert x[100] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(398.0)

    def test_average_beat_axis_uses_sample_spacing(self):
        lead = _make_ecg_lead(fs=500, duration=5.0)
        peaks = np.array([400, 800, 1200, 1600])
        x = plot_average_beat(lead, peaks=peaks, show=False).axes[0].get_lines()[0].get_xdata()
        assert x[0] == pytest.approx(-200.0)
        assert np.diff(x) == pytest.approx(np.full(len(x) - 1, 2.0))

    def test_zero_sampling_rate_is_a_clear_error(self):
        lead = Lead(label="I", samples=np.zeros(10), sampling_rate=0)
        with pytest.raises(ValueError, match="no sampling rate"):
            plot_lead(lead, show=False)
        plot_lead(lead, show=False, x_axis="samples")

    def test_empty_lead(self):
        fig = plot_lead(Lead(label="I", samples=np.zeros(0), sampling_rate=500), show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_grid_follows_paper(self):
        ax = plot_lead(_make_lead(duration=2.0), show=False, show_grid=True).axes[0]
        ticks = ax.get_xticks()
        assert np.diff(ticks) == pytest.approx(np.full(len(ticks) - 1, 0.2))
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels[int(np.argmin(abs(ticks - 1.0)))] == "1"
        assert labels[int(np.argmin(abs(ticks - 0.2)))] == ""
        assert np.diff(ax.get_yticks())[0] == pytest.approx(0.5)


class TestMixedLengths:
    def _leads(self):
        return [
            Lead(label="I", samples=np.zeros(5000), sampling_rate=500, units="mV"),
            Lead(label="II", samples=np.zeros(10000), sampling_rate=1000, units="mV"),
            Lead(label="III", samples=np.zeros(2500), sampling_rate=500, units="mV"),
        ]

    @pytest.mark.parametrize("func", [plot_leads, plot_12lead])
    def test_shared_axis_covers_longest_lead(self, func):
        fig = func(self._leads(), show=False)
        for ax in fig.axes:
            assert ax.get_xlim() == pytest.approx((0.0, 9.999))

    def test_unshared_axes_fit_each_lead(self):
        fig = plot_leads(self._leads(), show=False, share_x=False)
        assert fig.axes[2].get_xlim() == pytest.approx((0.0, 4.998))


class TestUnitLabels:
    def test_single_lead_units(self):
        assert plot_lead(_make_lead(), show=False).axes[0].get_ylabel() == "Amplitude (mV)"
        assert plot_lead(_raw_lead(), show=False).axes[0].get_ylabel() == "Amplitude (raw counts)"
        assert plot_lead(np.zeros(100), fs=100, show=False).axes[0].get_ylabel() == "Amplitude"

    def test_multi_lead_units(self):
        fig = plot_leads([_make_lead("I"), _raw_lead("II")], show=False)
        assert [ax.get_ylabel() for ax in fig.axes] == ["I\n(mV)", "II\n(raw counts)"]
        fig = plot_12lead([_make_lead("I"), _raw_lead("II")], show=False)
        assert [ax.get_title(loc="left") for ax in fig.axes] == ["I (mV)", "II (raw counts)"]

    def test_lead_objects_keep_their_labels(self):
        leads = [_make_lead("Lead 1"), _make_lead("Lead 2")]
        plot_12lead(leads, show=False)
        assert [ld.label for ld in leads] == ["Lead 1", "Lead 2"]

    def test_arrays_get_standard_names(self):
        fig = plot_12lead(np.zeros((3, 500)), fs=500, show=False)
        assert [ax.get_title(loc="left") for ax in fig.axes] == ["I", "II", "III"]


class TestReportWindow:
    def test_columns_show_consecutive_segments(self):
        record = _make_record(duration=12.0)
        fig = plot_report(record, show=False)
        cells = {ax.get_title(loc="left"): ax for ax in fig.axes}
        assert cells["I (mV)"].get_xlim() == pytest.approx((0.0, 2.5))
        assert cells["aVR (mV)"].get_xlim() == pytest.approx((2.5, 5.0))
        assert cells["V4 (mV)"].get_xlim() == pytest.approx((7.5, 10.0))
        x = cells["II rhythm strip (mV)"].get_lines()[0].get_xdata()
        assert x[-1] < 10.0


class TestStyleAndDependencies:
    def test_global_style_untouched(self):
        before = dict(matplotlib.rcParams)
        plot_lead(_make_lead(), show=False)
        assert matplotlib.rcParams["axes.grid"] == before["axes.grid"]
        assert matplotlib.rcParams["axes.edgecolor"] == before["axes.edgecolor"]

    @pytest.mark.parametrize("func, args", [
        (plot_lead, (np.zeros(10),)),
        (plot_poincare, (np.array([800.0, 810.0, 790.0]),)),
        (plot_hrv_summary, (np.array([800.0, 810.0, 790.0, 805.0]),)),
    ])
    def test_missing_matplotlib(self, monkeypatch, func, args):
        import sys

        monkeypatch.setitem(sys.modules, "matplotlib", None)
        with pytest.raises(ImportError, match=r"ecgdatakit\[plotting\]"):
            func(*args, fs=10, show=False) if func is plot_lead else func(*args, show=False)


class TestLongAndRR:
    def _long(self, label="II"):
        samples = np.zeros(500_000)
        samples[123_457] = 4.0
        return Lead(label=label, samples=samples, sampling_rate=500, units="mV")

    @pytest.mark.parametrize("func", [plot_lead, plot_leads, plot_12lead])
    def test_long_leads_are_decimated_with_warning(self, func):
        arg = self._long() if func is plot_lead else [self._long("I"), self._long("II")]
        with pytest.warns(UserWarning, match="min/max decimation") as rec:
            fig = func(arg, show=False)
        assert rec[0].filename == __file__
        line = fig.axes[0].get_lines()[0]
        assert len(line.get_ydata()) <= 200_002
        assert line.get_ydata().max() == 4.0
        assert line.get_xdata()[-1] == pytest.approx(499_999 / 500)

    def test_nan_rr_is_a_clear_error(self):
        rr = np.array([800.0, np.nan, 790.0, 805.0])
        for func in (plot_hrv_summary, plot_poincare, plot_rr_tachogram):
            with pytest.raises(ValueError, match="NaN"):
                func(rr, show=False)

    def test_two_rr_values_do_not_warn(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            plot_hrv_summary(np.array([800.0, 810.0]), show=False)

    def test_frequency_panel_matches_hrv_module(self):
        from ecgdatakit.processing.hrv import frequency_domain

        rng = np.random.default_rng(0)
        rr = 800 + 40 * np.sin(np.arange(400) * 0.6) + rng.normal(0, 5, 400)
        fig = plot_hrv_summary(rr, show=False)
        heights = [p.get_height() for p in fig.axes[2].patches]
        fd = frequency_domain(rr)
        assert heights == pytest.approx([fd["vlf_power"], fd["lf_power"], fd["hf_power"]])

    def test_quality_on_a_window(self):
        from ecgdatakit.processing.quality import signal_quality_index

        lead = _make_ecg_lead(duration=20.0)
        fig = plot_quality([lead], show=False, seconds=5.0)
        expected = signal_quality_index(Lead(label="II", samples=lead.samples[:2500],
                                             sampling_rate=500, units="mV"))
        assert fig.axes[0].patches[0].get_height() == pytest.approx(expected)
        assert "first 5 s" in fig.axes[0].get_title()
