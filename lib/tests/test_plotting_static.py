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
    return Lead(label=label, samples=samples.astype(np.float64), sample_rate=fs, units="mV")


def _make_ecg_lead(label: str = "II", fs: int = 500, duration: float = 5.0) -> Lead:
    """Create a synthetic lead with QRS-like peaks for R-peak detection."""
    n = int(fs * duration)
    t = np.arange(n, dtype=np.float64) / fs
    signal = np.zeros(n, dtype=np.float64)
    # Add QRS complexes every ~0.8 seconds (75 bpm)
    rr_samples = int(0.8 * fs)
    for pos in range(rr_samples, n, rr_samples):
        lo = max(0, pos - int(0.02 * fs))
        hi = min(n, pos + int(0.02 * fs))
        for j in range(lo, hi):
            signal[j] = 1.5 * np.exp(-((j - pos) / (0.008 * fs)) ** 2)
    signal += 0.05 * np.random.default_rng(42).standard_normal(n)
    return Lead(label=label, samples=signal, sample_rate=fs, units="mV")


def _make_12leads(fs: int = 500, duration: float = 5.0) -> list[Lead]:
    """Create all 12 leads."""
    labels = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    return [_make_ecg_lead(lbl, fs, duration) for lbl in labels]


def _make_record(fs: int = 500, duration: float = 5.0) -> ECGRecord:
    """Create a full ECGRecord with metadata."""
    rec = RecordingInfo()
    rec.acquisition.signal.sample_rate = fs
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
        fig = plot_hrv_summary(rr)
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
