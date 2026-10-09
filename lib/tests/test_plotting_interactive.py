"""Tests for ecgdatakit.plotting.interactive (plotly plots)."""

from __future__ import annotations

import numpy as np
import pytest

from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    GlobalMeasurements,
    Interpretation,
    Lead,
    PatientInfo,
    RecordingInfo,
)

plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402

from ecgdatakit.plotting.interactive import (  # noqa: E402
    iplot_12lead,
    iplot_lead,
    iplot_leads,
    iplot_peaks,
    iplot_poincare,
    iplot_report,
    iplot_rr_tachogram,
    iplot_spectrum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_lead(label: str = "II", fs: int = 500, duration: float = 5.0) -> Lead:
    n = int(fs * duration)
    t = np.arange(n, dtype=np.float64) / fs
    samples = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.default_rng(42).standard_normal(n)
    return Lead(label=label, samples=samples.astype(np.float64), sampling_rate=fs, units="mV")


def _make_ecg_lead(label: str = "II", fs: int = 500, duration: float = 5.0) -> Lead:
    n = int(fs * duration)
    signal = np.zeros(n, dtype=np.float64)
    rr_samples = int(0.8 * fs)
    for pos in range(rr_samples, n, rr_samples):
        lo = max(0, pos - int(0.02 * fs))
        hi = min(n, pos + int(0.02 * fs))
        for j in range(lo, hi):
            signal[j] = 1.5 * np.exp(-((j - pos) / (0.008 * fs)) ** 2)
    signal += 0.05 * np.random.default_rng(42).standard_normal(n)
    return Lead(label=label, samples=signal, sampling_rate=fs, units="mV")


def _make_12leads(fs: int = 500, duration: float = 5.0) -> list[Lead]:
    labels = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    return [_make_ecg_lead(lbl, fs, duration) for lbl in labels]


def _make_record(fs: int = 500, duration: float = 5.0) -> ECGRecord:
    rec = RecordingInfo()
    rec.acquisition.signal.sampling_rate = fs
    rec.device = DeviceInfo(manufacturer="TestCo", model="ECG-2000")
    return ECGRecord(
        patient=PatientInfo(patient_id="P001", first_name="Jane", last_name="Doe", age=45, sex="F"),
        recording=rec,
        leads=_make_12leads(fs, duration),
        interpretation=Interpretation(statements=[("Normal sinus rhythm", "")], severity="NORMAL"),
        measurements=GlobalMeasurements(heart_rate=75, pr_interval=160, qrs_duration=90, qtc_bazett=410),
        source_format="test",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIplotLead:
    def test_returns_go_figure(self):
        fig = iplot_lead(_make_lead())
        assert isinstance(fig, go.Figure)

    def test_with_peaks(self):
        peaks = np.array([100, 500, 900], dtype=np.intp)
        fig = iplot_lead(_make_lead(), peaks=peaks)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # signal + peaks

    def test_without_peaks(self):
        fig = iplot_lead(_make_lead())
        assert len(fig.data) == 1  # signal only


class TestIplotLeads:
    def test_returns_go_figure(self):
        leads = [_make_lead("I"), _make_lead("II"), _make_lead("III")]
        fig = iplot_leads(leads)
        assert isinstance(fig, go.Figure)

    def test_accepts_ecgrecord(self):
        fig = iplot_leads(_make_record())
        assert isinstance(fig, go.Figure)

    def test_empty_leads(self):
        fig = iplot_leads([])
        assert isinstance(fig, go.Figure)

    def test_with_peaks_dict(self):
        leads = [_make_lead("I"), _make_lead("II")]
        peaks_dict = {"I": np.array([100, 500], dtype=np.intp)}
        fig = iplot_leads(leads, peaks_dict=peaks_dict)
        assert isinstance(fig, go.Figure)


class TestIplot12Lead:
    def test_returns_go_figure(self):
        leads = _make_12leads()
        fig = iplot_12lead(leads)
        assert isinstance(fig, go.Figure)

    def test_with_record(self):
        record = _make_record()
        fig = iplot_12lead(record)
        assert isinstance(fig, go.Figure)


class TestIplotPeaks:
    def test_returns_go_figure(self):
        lead = _make_ecg_lead()
        fig = iplot_peaks(lead)
        assert isinstance(fig, go.Figure)

    def test_with_explicit_peaks(self):
        lead = _make_lead()
        peaks = np.array([100, 500, 900], dtype=np.intp)
        fig = iplot_peaks(lead, peaks=peaks)
        assert isinstance(fig, go.Figure)


class TestIplotSpectrum:
    def test_welch(self):
        fig = iplot_spectrum(_make_lead(), method="welch")
        assert isinstance(fig, go.Figure)

    def test_fft(self):
        fig = iplot_spectrum(_make_lead(), method="fft")
        assert isinstance(fig, go.Figure)


class TestIplotRRTachogram:
    def test_returns_go_figure(self):
        rr = np.array([800, 810, 790, 820, 780, 800, 815], dtype=np.float64)
        fig = iplot_rr_tachogram(rr)
        assert isinstance(fig, go.Figure)


class TestIplotPoincare:
    def test_returns_go_figure(self):
        rr = np.array([800, 810, 790, 820, 780, 800, 815], dtype=np.float64)
        fig = iplot_poincare(rr)
        assert isinstance(fig, go.Figure)

    def test_short_rr(self):
        rr = np.array([800], dtype=np.float64)
        fig = iplot_poincare(rr)
        assert isinstance(fig, go.Figure)


class TestIplotReport:
    def test_returns_go_figure(self):
        record = _make_record()
        fig = iplot_report(record)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Axes, units, long recordings and dependencies
# ---------------------------------------------------------------------------


class TestInteractiveAxes:
    def test_time_axis(self):
        fig = iplot_lead(_make_lead(fs=250, duration=4.0), show=False)
        x = np.asarray(fig.data[0].x)
        assert x[1] == pytest.approx(1 / 250) and x[-1] == pytest.approx(999 / 250)
        assert fig.layout.yaxis.title.text == "Amplitude (mV)"

    def test_mixed_lengths_range(self):
        leads = [
            Lead(label="I", samples=np.zeros(5000), sampling_rate=500, units="mV"),
            Lead(label="II", samples=np.zeros(2500), sampling_rate=500, units="mV"),
        ]
        fig = iplot_leads(leads, show=False)
        assert tuple(fig.layout.xaxis.range) == pytest.approx((0.0, 9.998))
        assert fig.layout.yaxis.title.text == "mV"

    def test_long_lead_is_decimated_with_warning(self):
        n = 500_000
        samples = np.zeros(n)
        samples[123_457] = 5.0
        lead = Lead(label="II", samples=samples, sampling_rate=500, units="mV")
        with pytest.warns(UserWarning, match="min/max decimation") as rec:
            fig = iplot_lead(lead, show=False)
        assert rec[0].filename == __file__
        y = np.asarray(fig.data[0].y)
        assert len(y) <= 200_002 and y.max() == 5.0
        assert np.asarray(fig.data[0].x)[-1] == pytest.approx((n - 1) / 500)

    def test_header_text_is_escaped(self):
        record = _make_record()
        record.patient.first_name = "<i>A</i>"
        fig = iplot_12lead(record, record=record, show=False)
        assert "&lt;i&gt;A&lt;/i&gt;" in fig.layout.annotations[-1].text

    def test_missing_plotly(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "plotly", None)
        with pytest.raises(ImportError, match=r"ecgdatakit\[plotting-interactive\]"):
            iplot_rr_tachogram(np.array([800.0, 810.0]), show=False)
