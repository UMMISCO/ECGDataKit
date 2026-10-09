"""Tests for the Parser framework and FileParser auto-discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecgdatakit.exceptions import UnsupportedFormatError
from ecgdatakit.parsing.parser import FileParser, Parser
from ecgdatakit.parsing.parsers.hl7_aecg import HL7aECGParser
from ecgdatakit.parsing.parsers.sierra_xml import SierraXMLParser
from ecgdatakit.parsing.parsers.ishne_holter import ISHNEHolterParser
from ecgdatakit.parsing.parsers.mortara_el250 import MortaraEL250Parser
from ecgdatakit.parsing.parsers.edf import EDFParser
from ecgdatakit.parsing.parsers.scp_ecg import SCPECGParser
from ecgdatakit.parsing.parsers.ge_muse_xml import GEMuseXMLParser
from ecgdatakit.parsing.parsers.wfdb import WFDBParser
from ecgdatakit.parsing.parsers.mfer import MFERParser


class TestParserABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Parser()


class TestFileParserDiscovery:
    def test_discovers_all_parsers(self):
        fp = FileParser()
        names = {p.__name__ for p in fp.parsers}
        assert "HL7aECGParser" in names
        assert "SierraXMLParser" in names
        assert "ISHNEHolterParser" in names
        assert "MortaraEL250Parser" in names
        assert "EDFParser" in names
        assert "SCPECGParser" in names
        assert "GEMuseXMLParser" in names
        assert "WFDBParser" in names
        assert "MFERParser" in names
        assert "DICOMWaveformParser" in names
        assert "EDANARCHolterParser" in names
        assert "AliveCorKardiaParser" in names
        # Removed from the release: no public spec and no real sample
        assert "BeneHeartR12Parser" not in names
        assert "GEMAC2000Parser" not in names

    def test_discovers_correct_count(self):
        # 12 parsers (DICOM is discovered even without pydicom installed)
        assert len(FileParser().parsers) == 12

    def test_discovers_each_parser_once(self):
        names = [p.__name__ for p in FileParser().parsers]
        assert len(names) == len(set(names))

    def test_all_discovered_are_parser_subclasses(self):
        fp = FileParser()
        for p in fp.parsers:
            assert issubclass(p, Parser)

    def test_raises_on_missing_file(self):
        fp = FileParser()
        with pytest.raises(FileNotFoundError):
            fp.parse("/nonexistent/file.xml")

    def test_raises_on_unknown_format(self, tmp_path: Path):
        unknown = tmp_path / "data.xyz"
        unknown.write_text("not an ecg file")
        fp = FileParser()
        with pytest.raises(ValueError, match="No parser found"):
            fp.parse(unknown)
        with pytest.raises(UnsupportedFormatError):
            fp.parse(unknown)


class TestCanParse:
    # --- Original 4 parsers ---

    def test_hl7_aecg_detects_xml(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert HL7aECGParser.can_parse(hl7_aecg_file, header) is True

    def test_hl7_aecg_rejects_sierra(self, tmp_path: Path):
        f = tmp_path / "sierra.xml"
        f.write_text('<?xml version="1.0"?><restingecgdata></restingecgdata>')
        header = f.read_bytes()[:4096]
        assert HL7aECGParser.can_parse(f, header) is False

    def test_sierra_detects_xml(self, tmp_path: Path):
        f = tmp_path / "sierra.xml"
        f.write_text('<?xml version="1.0"?><restingecgdata></restingecgdata>')
        header = f.read_bytes()[:4096]
        assert SierraXMLParser.can_parse(f, header) is True

    def test_sierra_rejects_hl7(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert SierraXMLParser.can_parse(hl7_aecg_file, header) is False

    def test_ishne_detects_binary(self, ishne_file: Path):
        header = ishne_file.read_bytes()[:4096]
        assert ISHNEHolterParser.can_parse(ishne_file, header) is True

    def test_ishne_rejects_xml(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert ISHNEHolterParser.can_parse(hl7_aecg_file, header) is False

    def test_mortara_detects_xml(self, mortara_file: Path):
        header = mortara_file.read_bytes()[:4096]
        assert MortaraEL250Parser.can_parse(mortara_file, header) is True

    def test_mortara_rejects_hl7(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert MortaraEL250Parser.can_parse(hl7_aecg_file, header) is False

    # --- New parsers: EDF ---

    def test_edf_detects_binary(self, edf_file: Path):
        header = edf_file.read_bytes()[:4096]
        assert EDFParser.can_parse(edf_file, header) is True

    def test_edf_rejects_xml(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert EDFParser.can_parse(hl7_aecg_file, header) is False

    # --- SCP-ECG ---

    def test_scp_ecg_detects_binary(self, scp_ecg_file: Path):
        header = scp_ecg_file.read_bytes()[:4096]
        assert SCPECGParser.can_parse(scp_ecg_file, header) is True

    def test_scp_ecg_rejects_xml(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert SCPECGParser.can_parse(hl7_aecg_file, header) is False

    # --- GE MUSE XML ---

    def test_ge_muse_detects_xml(self, ge_muse_xml_file: Path):
        header = ge_muse_xml_file.read_bytes()[:4096]
        assert GEMuseXMLParser.can_parse(ge_muse_xml_file, header) is True

    def test_ge_muse_rejects_sierra(self, tmp_path: Path):
        f = tmp_path / "sierra.xml"
        f.write_text('<?xml version="1.0"?><restingecgdata></restingecgdata>')
        header = f.read_bytes()[:4096]
        assert GEMuseXMLParser.can_parse(f, header) is False

    def test_ge_muse_rejects_mortara(self, mortara_file: Path):
        header = mortara_file.read_bytes()[:4096]
        assert GEMuseXMLParser.can_parse(mortara_file, header) is False

    # --- WFDB ---

    def test_wfdb_detects_hea(self, wfdb_file: Path):
        header = wfdb_file.read_bytes()[:4096]
        assert WFDBParser.can_parse(wfdb_file, header) is True

    def test_wfdb_rejects_non_hea(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert WFDBParser.can_parse(hl7_aecg_file, header) is False

    # --- MFER ---

    def test_mfer_detects_binary(self, mfer_file: Path):
        header = mfer_file.read_bytes()[:4096]
        assert MFERParser.can_parse(mfer_file, header) is True

    def test_mfer_rejects_xml(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert MFERParser.can_parse(hl7_aecg_file, header) is False

class TestCrossDetection:
    """Verify that each fixture file is only claimed by its correct parser."""

    def _matching_parsers(self, file_path: Path) -> list[str]:
        header = file_path.read_bytes()[:4096] if file_path.is_file() else b""
        return [
            p.__name__ for p in FileParser().parsers
            if p.can_parse(file_path, header)
        ]

    def test_hl7_exclusive(self, hl7_aecg_file: Path):
        matches = self._matching_parsers(hl7_aecg_file)
        assert matches == ["HL7aECGParser"]

    def test_mortara_exclusive(self, mortara_file: Path):
        matches = self._matching_parsers(mortara_file)
        assert matches == ["MortaraEL250Parser"]

    def test_ishne_exclusive(self, ishne_file: Path):
        matches = self._matching_parsers(ishne_file)
        assert matches == ["ISHNEHolterParser"]

    def test_edf_exclusive(self, edf_file: Path):
        matches = self._matching_parsers(edf_file)
        assert matches == ["EDFParser"]

    def test_scp_ecg_exclusive(self, scp_ecg_file: Path):
        matches = self._matching_parsers(scp_ecg_file)
        assert matches == ["SCPECGParser"]

    def test_ge_muse_exclusive(self, ge_muse_xml_file: Path):
        matches = self._matching_parsers(ge_muse_xml_file)
        assert matches == ["GEMuseXMLParser"]

    def test_wfdb_exclusive(self, wfdb_file: Path):
        matches = self._matching_parsers(wfdb_file)
        assert matches == ["WFDBParser"]

    def test_mfer_exclusive(self, mfer_file: Path):
        matches = self._matching_parsers(mfer_file)
        assert matches == ["MFERParser"]

    def test_dicom_exclusive(self, dicom_file: Path):
        assert self._matching_parsers(dicom_file) == ["DICOMWaveformParser"]

    def test_edan_files_and_directory_exclusive(self, edan_arc_dir: Path):
        for path in (edan_arc_dir, edan_arc_dir.parent / "ecgraw.dat", edan_arc_dir.parent):
            assert self._matching_parsers(path) == ["EDANARCHolterParser"]

    def test_edan_archives_exclusive(self, edan_arc_archive: Path, neutral_holter_arc_file: Path):
        assert self._matching_parsers(edan_arc_archive) == ["EDANARCHolterParser"]
        assert self._matching_parsers(neutral_holter_arc_file) == ["EDANARCHolterParser"]

    def test_wfdb_dat_exclusive(self, wfdb_file: Path):
        assert self._matching_parsers(wfdb_file.with_suffix(".dat")) == ["WFDBParser"]

    def test_bdf_claimed_by_edf_parser(self, tmp_path: Path):
        p = tmp_path / "x.bdf"
        p.write_bytes(b"\xffBIOSEMI" + b" " * 248)
        assert self._matching_parsers(p) == ["EDFParser"]
