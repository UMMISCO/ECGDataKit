"""Tests for the AliveCor Kardia JSON parser.

Files follow the "Single ECG" example of the Kardia API documentation
(https://developers.kardia.com).
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit import FileParser
from ecgdatakit.exceptions import CorruptedFileError, UnsupportedFormatError
from ecgdatakit.parsing.parsers.alivecor_kardia import AliveCorKardiaParser

LEADS = ("leadI", "leadII", "leadIII", "AVR", "AVL", "AVF")


def _block(values: dict[str, list[int]], **extra) -> dict:
    block = {"frequency": 300, "mainsFrequency": 50, "samples": values,
             "amplitudeResolution": 500, "numLeads": len(values)}
    block.update(extra)
    return block


def _recording(n_leads: int = 6, **overrides) -> dict:
    raw = {k: [100 * (i + 1), -200, 0, 4000] for i, k in enumerate(LEADS[:n_leads])}
    enhanced = {k: [v + 1 for v in s] for k, s in raw.items()}
    doc = {
        "id": "epgstuys6lpj0dssj2tjyia1v",
        "patientID": "DwftkmZIYHm7ysxEMAhJcap1x38vz2un",
        "heartRate": 89,
        "duration": 30000,
        "recordedAt": "2021-04-16T15:19:09+05:30",
        "algorithmDetermination": "normal",
        "data": {"raw": _block(raw), "enhanced": _block(enhanced)},
        "deviceInfo": {"hardwareType": "kardia_6l", "hardwareRevision": "X08",
                       "firmwareRevision": "1.3.4", "serialNumber": "20042543K"},
        "tags": [],
        "qtMeasurement": {"qt": {"Value": 354}, "rr": {"Value": 666.667},
                          "qtcb": {"Value": 434}, "qtcf": {"Value": 405}},
    }
    doc.update(overrides)
    return doc


def _write(tmp_path: Path, doc: dict, name: str = "rec.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class TestDetection:
    def test_detects_kardia_json(self, tmp_path: Path):
        path = _write(tmp_path, _recording())
        assert AliveCorKardiaParser.can_parse(path, path.read_bytes()[:4096])

    def test_rejects_other_json(self, tmp_path: Path):
        path = _write(tmp_path, {"recordedAt": "2021-01-01", "name": "x"})
        assert not AliveCorKardiaParser.can_parse(path, path.read_bytes()[:4096])

    def test_rejects_xml(self, tmp_path: Path):
        header = b'<?xml version="1.0"?><ECG recordedAt="x" patientID="y"/>'
        assert not AliveCorKardiaParser.can_parse(tmp_path / "a.xml", header)


class TestSignal:
    def test_raw_and_enhanced_in_mv(self, tmp_path: Path):
        record = FileParser().parse(_write(tmp_path, _recording()))
        assert [lead.label for lead in record.leads] == ["I", "II", "III", "aVR", "aVL", "aVF"]
        assert [lead.label for lead in record.leads_enhanced] == [lead.label for lead in record.leads]
        # Documented conversion: mV = samples / (1e6 / amplitudeResolution)
        np.testing.assert_allclose(record.leads[0].samples, np.array([100, -200, 0, 4000]) / 2000)
        np.testing.assert_allclose(record.leads_enhanced[0].samples,
                                   np.array([101, -199, 1, 4001]) / 2000)
        assert all(lead.units == "mV" for lead in record.leads + record.leads_enhanced)
        assert record.leads[0].sampling_rate == 300

    def test_raw_counts_without_auto_scale(self, tmp_path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = FileParser().parse(_write(tmp_path, _recording()), auto_scale=False)
        lead = record.leads[0]
        np.testing.assert_array_equal(lead.samples, [100, -200, 0, 4000])
        assert lead.is_raw and lead.resolution == 0.5 and lead.resolution_unit == "uV"
        assert lead.adc_resolution == 500 and lead.adc_resolution_unit == "nV"
        assert record.leads_enhanced[0].is_raw

    def test_single_lead(self, tmp_path: Path):
        record = FileParser().parse(_write(tmp_path, _recording(n_leads=1)))
        assert [lead.label for lead in record.leads] == ["I"]

    def test_without_enhanced(self, tmp_path: Path):
        doc = _recording()
        del doc["data"]["enhanced"]
        record = FileParser().parse(_write(tmp_path, doc))
        assert record.leads_enhanced == []

    def test_missing_resolution_stays_raw(self, tmp_path: Path):
        doc = _recording()
        del doc["data"]["raw"]["amplitudeResolution"]
        with pytest.warns(UserWarning, match="amplitudeResolution"):
            record = FileParser().parse(_write(tmp_path, doc))
        assert record.leads[0].is_raw and record.leads[0].resolution_unit == ""

    def test_missing_frequency_is_corrupted(self, tmp_path: Path):
        doc = _recording()
        del doc["data"]["raw"]["frequency"]
        with pytest.raises(CorruptedFileError, match="frequency"):
            FileParser().parse(_write(tmp_path, doc))

    def test_num_leads_mismatch_warns(self, tmp_path: Path):
        doc = _recording()
        doc["data"]["raw"]["numLeads"] = 1
        with pytest.warns(UserWarning, match="numLeads"):
            FileParser().parse(_write(tmp_path, doc))

    def test_json_export_and_units(self, tmp_path: Path):
        record = FileParser().parse(_write(tmp_path, _recording()))
        data = json.loads(record.to_json())
        assert len(data["leads_enhanced"]) == 6
        assert data["leads_enhanced"][0]["samples"][0] == pytest.approx(0.0505)
        uv = record.convert_units("uV")
        assert uv.leads_enhanced[0].samples[0] == pytest.approx(50.5)


class TestMetadata:
    def test_fields(self, tmp_path: Path):
        record = FileParser().parse(_write(tmp_path, _recording()))
        assert record.source_format == "alivecor_kardia"
        assert record.patient.patient_id == "DwftkmZIYHm7ysxEMAhJcap1x38vz2un"
        assert record.recording.date == datetime(
            2021, 4, 16, 15, 19, 9, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        assert record.recording.duration == timedelta(seconds=30)
        device = record.recording.device
        assert (device.manufacturer, device.model, device.serial_number,
                device.software_version) == ("AliveCor", "kardia_6l", "20042543K", "1.3.4")
        assert record.annotations["hardware_revision"] == "X08"
        assert record.annotations["mains_frequency_hz"] == "50"
        assert record.annotations["recording_id"] == "epgstuys6lpj0dssj2tjyia1v"

    def test_measurements_not_repeated_in_annotations(self, tmp_path: Path):
        record = FileParser().parse(_write(tmp_path, _recording()))
        m = record.measurements
        assert (m.heart_rate, m.qt_interval, m.rr_interval, m.qtc_bazett,
                m.qtc_fridericia) == (89, 354, 667, 434, 405)
        assert not any(k.startswith("qt") for k in record.annotations)

    def test_machine_interpretation(self, tmp_path: Path):
        record = FileParser().parse(_write(tmp_path, _recording(algorithmDetermination="afib")))
        assert record.interpretation.statements == [("afib", "")]
        assert record.interpretation.source == "machine"

    def test_kardia_12l_unsupported(self, tmp_path: Path):
        doc = {"id": "x", "recordedAt": "2024-04-04T10:26:33-05:30",
               "algorithmDetermination": "normal", "data12L": {}}
        with pytest.raises(UnsupportedFormatError, match="12L"):
            FileParser().parse(_write(tmp_path, doc))

    def test_malformed_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text('{"recordedAt": "x", "patientID": "y", ', encoding="utf-8")
        with pytest.raises(CorruptedFileError):
            FileParser().parse(path)
