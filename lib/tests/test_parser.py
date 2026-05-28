"""Tests for the Parser framework and FileParser auto-discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

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
from ecgdatakit.parsing.parsers.beneheart_r12 import BeneHeartR12Parser
from ecgdatakit.parsing.parsers.ge_mac2000 import GEMAC2000Parser


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
        assert "BeneHeartR12Parser" in names
        assert "GEMAC2000Parser" in names

    def test_discovers_correct_count(self):
        fp = FileParser()
        # 4 original + 8 new = 12 parsers total
        # (DICOM parser is also discovered even without pydicom installed)
        assert len(fp.parsers) >= 11

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

    # --- BeneHeart R12 ---

    def test_beneheart_detects_xml(self, beneheart_r12_file: Path):
        header = beneheart_r12_file.read_bytes()[:4096]
        assert BeneHeartR12Parser.can_parse(beneheart_r12_file, header) is True

    def test_beneheart_rejects_hl7(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert BeneHeartR12Parser.can_parse(hl7_aecg_file, header) is False

    # --- GE MAC 2000 ---

    def test_ge_mac2000_detects_xml(self, ge_mac2000_file: Path):
        header = ge_mac2000_file.read_bytes()[:4096]
        assert GEMAC2000Parser.can_parse(ge_mac2000_file, header) is True

    def test_ge_mac2000_rejects_muse(self, ge_muse_xml_file: Path):
        header = ge_muse_xml_file.read_bytes()[:4096]
        assert GEMAC2000Parser.can_parse(ge_muse_xml_file, header) is False

    def test_ge_mac2000_rejects_hl7(self, hl7_aecg_file: Path):
        header = hl7_aecg_file.read_bytes()[:4096]
        assert GEMAC2000Parser.can_parse(hl7_aecg_file, header) is False


class TestCrossDetection:
    """Verify that each fixture file is only claimed by its correct parser."""

    ALL_PARSERS = [
        HL7aECGParser, SierraXMLParser, ISHNEHolterParser, MortaraEL250Parser,
        EDFParser, SCPECGParser, GEMuseXMLParser,
        WFDBParser, MFERParser, BeneHeartR12Parser, GEMAC2000Parser,
    ]

    def _matching_parsers(self, file_path: Path) -> list[str]:
        header = file_path.read_bytes()[:4096]
        return [
            p.__name__ for p in self.ALL_PARSERS
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

    def test_beneheart_exclusive(self, beneheart_r12_file: Path):
        matches = self._matching_parsers(beneheart_r12_file)
        assert matches == ["BeneHeartR12Parser"]

    def test_ge_mac2000_exclusive(self, ge_mac2000_file: Path):
        matches = self._matching_parsers(ge_mac2000_file)
        assert matches == ["GEMAC2000Parser"]
