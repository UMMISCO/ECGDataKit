"""Tests for ecgdatakit.anonymize.

Each format is anonymized from the files built by the parser fixtures. The
anonymized file must be read by the same parser with the same samples, the
patient name and ID must hold only the pseudonym, and no identity value may
remain in the file or its name. Raw files must not change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("shortuuid")

from ecgdatakit import FileParser  # noqa: E402
from ecgdatakit.anonymize import COLUMNS, Anonymizer, DatasetLocked  # noqa: E402
from ecgdatakit.anonymize._filenames import new_file_name  # noqa: E402
from ecgdatakit.anonymize._identity import Identity, Replacer  # noqa: E402
from ecgdatakit.cli import main  # noqa: E402

FORMAT_FIXTURES = [
    "hl7_aecg_file", "mortara_file", "ishne_file", "edf_file", "scp_ecg_file",
    "ge_muse_xml_file", "dicom_12lead_file", "mfer_file",
]


def _parse(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FileParser().parse(path, auto_scale=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, files: list[Path], patient: str = "PATIENT-001",
            dataset: str = "dataset-a", visit: str = "V1") -> Path:
    """SOURCE/<dataset>/xml/RAW/<patient>/<visit>/<files>"""
    raw = tmp_path / "source" / dataset / "xml" / "RAW" / patient / visit
    raw.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, raw / f.name)
    return tmp_path / "source"


def _anonymizer(source: Path, **kwargs) -> Anonymizer:
    return Anonymizer(source, datasets=True, patients_dir_name="RAW", **kwargs)


def _rows(source: Path, dataset: str = "dataset-a") -> list[dict]:
    with open(source / dataset / "anonymization_catalog.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _identity_words(row: dict) -> list[str]:
    words = []
    for col in ("original_last_name", "original_first_name"):
        for value in row[col].split(" | "):
            words += [w for w in re.split(r"[\s,;^_.\-']+", value) if len(w) >= 3]
    for col in ("original_patient_id", "original_ecg_id"):
        words += [v for v in row[col].split(" | ") if len(v) >= 4]
    return words


class TestFormats:
    @pytest.mark.parametrize("fixture", FORMAT_FIXTURES)
    def test_format(self, fixture, request, tmp_path):
        src = request.getfixturevalue(fixture)
        base = _source(tmp_path, [src])
        raw_file = next((base / "dataset-a").rglob(src.name))
        raw_hash = _sha(raw_file)

        [report] = _anonymizer(base, threads=2).run()
        assert report.ok, report.errors
        [row] = _rows(base)
        assert row["status"] == "anonymized"
        assert row["folder_anonymized"] == "FALSE"
        assert _sha(raw_file) == raw_hash, "raw file modified"

        out = base / "dataset-a" / row["anonymized_path"]
        before, after = _parse(raw_file), _parse(out)
        assert after.file_format.parser == before.file_format.parser
        for a, b in zip(before.leads, after.leads):
            np.testing.assert_array_equal(a.samples, b.samples)
        code = row["patient_code"]
        assert after.patient.patient_id in ("", code)
        assert after.patient.last_name in ("", code)
        assert after.patient.first_name == ""
        assert after.patient.birth_date == before.patient.birth_date

        text = out.read_bytes().decode("latin-1").lower() + "\n" + out.stem.lower()
        for word in _identity_words(row):
            assert not re.search(rf"(?<![^\W_]){re.escape(word.lower())}(?![^\W_])", text), \
                f"{word!r} left in the anonymized file"

    def test_ishne_checksum_recomputed(self, ishne_file, tmp_path):
        base = _source(tmp_path, [ishne_file])
        _anonymizer(base).run()
        [row] = _rows(base)
        out = base / "dataset-a" / row["anonymized_path"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a CRC mismatch would warn
            record = FileParser().parse(out, auto_scale=False)
        assert record.raw_metadata.get("checksum_valid") in (True, None)

    def test_scp_crc_and_pointers_updated(self, scp_ecg_file, tmp_path):
        base = _source(tmp_path, [scp_ecg_file])
        _anonymizer(base).run()
        [row] = _rows(base)
        record = _parse(base / "dataset-a" / row["anonymized_path"])
        assert record.raw_metadata["checksum_valid"] is True

    def test_xml_layout_kept(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file])
        _anonymizer(base).run()
        [row] = _rows(base)
        raw = hl7_aecg_file.read_bytes()
        out = (base / "dataset-a" / row["anonymized_path"]).read_bytes()
        # Same lines except those holding identity values
        changed = [i for i, (a, b) in enumerate(zip(raw.splitlines(), out.splitlines())) if a != b]
        assert len(raw.splitlines()) == len(out.splitlines())
        assert 0 < len(changed) <= 6

    def test_wfdb_comments_and_record_rename(self, tmp_path):
        record = tmp_path / "src"
        record.mkdir()
        (record / "Doe_rec.dat").write_bytes(np.arange(20, dtype="<i2").tobytes())
        (record / "Doe_rec.hea").write_text(
            "Doe_rec 1 100 20\n"
            "Doe_rec.dat 16 200 16 0 0 0 0 II\n"
            "# name: John Doe\n# id: PX-4521\n# age: 60\n# seen by John's doctor\n")
        base = _source(tmp_path, [record / "Doe_rec.hea", record / "Doe_rec.dat"])
        [report] = _anonymizer(base).run()
        assert report.ok, report.errors
        [row] = _rows(base)
        code = row["patient_code"]
        assert row["anonymized_file_name"] == f"{code}_rec.hea"
        out = base / "dataset-a" / row["anonymized_path"]
        assert (out.parent / f"{code}_rec.dat").exists()
        header = out.read_text()
        assert header.splitlines()[0] == f"{code}_rec 1 100 20"
        assert f"{code}_rec.dat 16" in header
        assert "Doe" not in header and "John" not in header and "PX-4521" not in header
        assert "# age: 60" in header
        assert len(_parse(out).leads[0].samples) == 20

    def test_alivecor(self, tmp_path):
        doc = {"id": "rec8f3kd92jd83", "patientID": "PAT7d9s8f7d9s8", "heartRate": 70,
               "duration": 1000, "recordedAt": "2021-04-16T15:19:09+05:30",
               "algorithmDetermination": "normal", "note": "follow-up",
               "data": {"raw": {"frequency": 300, "amplitudeResolution": 500, "numLeads": 1,
                                "samples": {"leadI": [1, 2, 3]}}}}
        src = tmp_path / "PAT7d9s8f7d9s8_rec.json"
        src.write_text(json.dumps(doc, indent=2))
        base = _source(tmp_path, [src])
        _anonymizer(base).run()
        [row] = _rows(base)
        out = base / "dataset-a" / row["anonymized_path"]
        assert row["anonymized_file_name"] == f"{row['patient_code']}_rec.json"
        new = json.loads(out.read_text())
        assert new["patientID"] == row["patient_code"] and new["id"] == row["ecg_code"]
        assert new["data"] == doc["data"] and new["note"] == "follow-up"


class TestNames:
    def _identity(self) -> Identity:
        i = Identity()
        i.add("last_names", "Doe")
        i.add("first_names", "Zoé")
        i.add("patient_ids", "PID 0042")
        i.add("ecg_ids", "ABC123XYZ")
        return i

    def test_parts_replaced_dates_and_visits_kept(self):
        name, risks = new_file_name("R^ECG^F^0^PID 0042^DOE_20240109083645_V1.xml",
                                    self._identity(), "PCODE", "ECODE", [])
        assert name == "R^ECG^F^0^PCODE^PCODE_20240109083645_V1.xml"
        assert risks == []

    def test_accents_and_case_ignored(self):
        name, _ = new_file_name("zoe_DOE_abc123xyz.ecg", self._identity(), "P", "E", [])
        assert name == "P_P_E.ecg"

    def test_initials_kept_and_reported(self):
        name, risks = new_file_name("DO_ZO_2024.xml", self._identity(), "P", "E", [])
        assert name == "DO_ZO_2024.xml"
        assert any("'DO'" in r for r in risks) and any("'ZO'" in r for r in risks)

    def test_folder_name_used_when_file_has_no_name(self):
        name, risks = new_file_name("2_ROE_Jane_01011970.ecg", Identity(), "P", "E",
                                    ["ROE Jane"])
        assert name == "2_P_P_01011970.ecg"
        assert risks

    def test_free_text_word_boundaries(self):
        r = Replacer(self._identity(), "P", "E")
        assert r.replace("Seen: Zoe DOE, Doerr not") == "Seen: P, Doerr not"
        # Short plain numbers that equal no whole ID are never touched
        assert r.replace_value("1234", data_like=True) == "1234"


class TestRuns:
    def test_counts_rerun_change_delete(self, hl7_aecg_file, mortara_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file, mortara_file])
        a = _anonymizer(base, threads=2)
        [r] = a.run()
        assert (r.found, r.anonymized, r.failed) == (2, 2, 0) and r.ok
        rows = {Path(x["raw_path"]).name: x for x in _rows(base)}
        assert rows["test_hl7.xml"]["patient_code"] == rows["test_mortara.xml"]["patient_code"]
        assert rows["test_hl7.xml"]["ecg_code"] != rows["test_mortara.xml"]["ecg_code"]

        [r] = a.run()
        assert (r.unchanged, r.anonymized, r.changed) == (2, 0, 0)

        raw_hl7 = base / "dataset-a" / rows["test_hl7.xml"]["raw_path"]
        raw_hl7.write_bytes(raw_hl7.read_bytes() + b"\n")
        (base / "dataset-a" / rows["test_mortara.xml"]["raw_path"]).unlink()
        [r] = a.run()
        assert (r.changed, r.deleted) == (1, 1) and r.ok
        after = {Path(x["raw_path"]).name: x for x in _rows(base)}
        assert after["test_hl7.xml"]["status"] == "changed"
        assert after["test_hl7.xml"]["last_change_at"]
        assert after["test_hl7.xml"]["patient_code"] == rows["test_hl7.xml"]["patient_code"]
        assert after["test_mortara.xml"]["status"] == "deleted"
        assert not (base / "dataset-a" / rows["test_mortara.xml"]["anonymized_path"]).exists()

    def test_patients_get_distinct_codes(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file], patient="A")
        _source(tmp_path, [hl7_aecg_file], patient="B")
        [r] = _anonymizer(base).run()
        assert r.ok
        codes = {x["patient_folder"]: x["patient_code"] for x in _rows(base)}
        assert len(codes) == 2 and len(set(codes.values())) == 2

    def test_unsupported_and_outputs_not_scanned(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file])
        folder = next((base / "dataset-a").rglob("V1"))
        (folder / "notes.pdf").write_bytes(b"%PDF-1.4 fake")
        (base / "dataset-a" / "xml" / "ANONYMIZED").mkdir()  # an old folder, left alone
        [r] = _anonymizer(base).run()
        assert (r.found, r.skipped_unsupported) == (1, 1)
        [r] = _anonymizer(base).run()
        assert r.found == 1  # the output folder is never read as raw files
        assert (base / "dataset-a" / "xml" / "ANONYMIZED").is_dir()
        assert not any((base / "dataset-a" / "ANONYMIZED").rglob("*.pdf"))

    def test_files_outside_raw_are_ignored(self, hl7_aecg_file, tmp_path):
        other = tmp_path / "source" / "dataset-a" / "PDF"
        other.mkdir(parents=True)
        shutil.copy2(hl7_aecg_file, other / "x.xml")
        [r] = _anonymizer(tmp_path / "source").run()
        assert r.found == 0

    def test_single_file_and_dry_run(self, hl7_aecg_file, mortara_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file, mortara_file])
        target = next((base / "dataset-a").rglob("test_hl7.xml"))
        [r] = _anonymizer(base, dry_run=True).run(paths=[target])
        assert r.found == 1 and not (base / "dataset-a" / "anonymization_catalog.csv").exists()
        [r] = _anonymizer(base).run(paths=[target])
        assert r.found == 1 and len(_rows(base)) == 1

    def test_catalog_columns(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file])
        _anonymizer(base).run()
        with open(base / "dataset-a" / "anonymization_catalog.csv", newline="") as f:
            assert next(csv.reader(f)) == COLUMNS
        [row] = _rows(base)
        assert row["original_patient_id"] == "SUBJ-001"
        assert row["original_last_name"] == "Doe"
        assert row["detected_at"] and row["anonymized_at"]
        assert len(row["raw_sha256"]) == 64

    def test_locked_source(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file])
        (base / "dataset-a" / ".anonymize.lock").write_text("other 1\n")
        with pytest.raises(DatasetLocked):
            _anonymizer(base).run()

    def test_failed_verification_publishes_nothing(self, hl7_aecg_file, tmp_path, monkeypatch):
        from ecgdatakit.anonymize import _engine
        from ecgdatakit.anonymize._verify import VerificationError

        def fail(*args, **kwargs):
            raise VerificationError("forced")

        monkeypatch.setattr(_engine, "verify", fail)
        base = _source(tmp_path, [hl7_aecg_file])
        [r] = _anonymizer(base).run()
        assert r.failed == 1
        [row] = _rows(base)
        assert row["status"] == "failed" and "forced" in row["message"]
        assert not any(p.is_file() for p in (base / "dataset-a" / "ANONYMIZED").rglob("*"))


class TestCommandLine:
    def test_run(self, hl7_aecg_file, tmp_path, capsys):
        base = _source(tmp_path, [hl7_aecg_file])
        assert main(["anonymize", "run", str(base), "--datasets", "--patients-dir-name", "RAW"]) == 0
        assert "1 anonymized" in capsys.readouterr().out

    def test_missing_path(self, tmp_path, capsys):
        (tmp_path / "source").mkdir()
        assert main(["anonymize", "run", str(tmp_path / "source"), str(tmp_path / "nope")]) == 2


class TestSourceModes:
    def test_source_is_one_dataset_without_patients_dir_name(self, hl7_aecg_file, mortara_file, tmp_path):
        source = tmp_path / "export"
        (source / "PATIENT-001").mkdir(parents=True)
        (source / "PATIENT-002" / "V2").mkdir(parents=True)
        shutil.copy2(hl7_aecg_file, source / "PATIENT-001" / hl7_aecg_file.name)
        shutil.copy2(mortara_file, source / "PATIENT-002" / "V2" / mortara_file.name)
        [r] = Anonymizer(source).run()
        assert r.ok and r.found == 2 and r.dataset == "export"
        with open(source / "anonymization_catalog.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        # Without patients_dir_name, patients follow their IDs, not folders
        assert len({x["patient_code"] for x in rows}) == 2
        assert {x["anonymized_path"].rsplit("/", 1)[0] for x in rows} == {
            "ANONYMIZED/PATIENT-001", "ANONYMIZED/PATIENT-002/V2"}
        # The output folder and the catalog are not read again as raw files
        [r] = Anonymizer(source).run()
        assert (r.found, r.unchanged) == (2, 2)

    def test_datasets_have_their_own_outputs(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file], dataset="dataset-a")
        _source(tmp_path, [hl7_aecg_file], dataset="dataset-b")
        reports = _anonymizer(base).run()
        assert sorted(r.dataset for r in reports) == ["dataset-a", "dataset-b"]
        for name in ("dataset-a", "dataset-b"):
            assert (base / name / "anonymization_catalog.csv").exists()
            assert (base / name / "ANONYMIZED").is_dir()
        [r] = _anonymizer(base).run(datasets=["dataset-b"])
        assert r.dataset == "dataset-b"

    def test_dataset_names_need_datasets_mode(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file])
        with pytest.raises(ValueError):
            Anonymizer(base).run(datasets=["dataset-a"])

    def test_single_file(self, hl7_aecg_file, mortara_file, tmp_path, capsys):
        folder = tmp_path / "inbox"
        folder.mkdir()
        target = folder / hl7_aecg_file.name
        shutil.copy2(hl7_aecg_file, target)
        shutil.copy2(mortara_file, folder / mortara_file.name)
        assert main(["anonymize", "run", str(target)]) == 0
        with open(folder / "anonymization_catalog.csv", newline="") as f:
            [row] = list(csv.DictReader(f))  # only the given file
        assert row["status"] == "anonymized" and row["raw_path"] == target.name
        assert (folder / row["anonymized_path"]).parent == folder / "ANONYMIZED"
        [r] = Anonymizer(target).run()
        assert r.unchanged == 1

    def test_single_file_rejects_datasets(self, hl7_aecg_file):
        with pytest.raises(ValueError):
            Anonymizer(hl7_aecg_file, datasets=True)

    def test_same_patient_id_in_two_folders_without_patients_dir(self, hl7_aecg_file, tmp_path):
        source = tmp_path / "export"
        for folder in ("a", "b/c"):
            (source / folder).mkdir(parents=True)
            shutil.copy2(hl7_aecg_file, source / folder / hl7_aecg_file.name)
        [r] = Anonymizer(source).run()
        assert r.ok and r.found == 2
        with open(source / "anonymization_catalog.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len({x["patient_code"] for x in rows}) == 1
        assert len({x["ecg_code"] for x in rows}) == 2


class TestECGCode:
    def test_ecg_code_is_a_uuid_written_as_is(self, hl7_aecg_file, tmp_path):
        import uuid as _uuid
        base = _source(tmp_path, [hl7_aecg_file])
        _anonymizer(base).run()
        [row] = _rows(base)
        code = row["ecg_code"]
        assert str(_uuid.UUID(code)) == code
        out = (base / "dataset-a" / row["anonymized_path"]).read_text()
        assert f'root="{code}"' in out
        assert "anonymized_ecg_id" not in row

    def test_dicom_gets_the_uuid_in_uid_form(self, dicom_12lead_file, tmp_path):
        import uuid as _uuid
        pydicom = pytest.importorskip("pydicom")
        base = _source(tmp_path, [dicom_12lead_file])
        _anonymizer(base).run()
        [row] = _rows(base)
        ds = pydicom.dcmread(base / "dataset-a" / row["anonymized_path"])
        assert ds.SOPInstanceUID == f"2.25.{_uuid.UUID(row['ecg_code']).int}"
        assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID


class TestPatientFolders:
    def test_off_by_default(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file], patient="DOE Ann")
        _anonymizer(base).run()
        [row] = _rows(base)
        assert row["folder_anonymized"] == "FALSE"
        assert "/DOE Ann/" in row["anonymized_path"]

    def test_patient_folder_renamed(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file], patient="DOE Ann", visit="V1")
        [r] = _anonymizer(base, anonymize_patient_folders=True).run()
        assert r.ok
        [row] = _rows(base)
        code = row["patient_code"]
        assert row["folder_anonymized"] == "TRUE"
        assert row["anonymized_path"].startswith(f"ANONYMIZED/xml/RAW/{code}/V1/")
        assert (base / "dataset-a" / row["anonymized_path"]).exists()
        assert not any((base / "dataset-a" / "ANONYMIZED").rglob("DOE Ann"))

    def test_sub_folder_holding_the_name_is_replaced(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file], patient="P1", visit="Doe V2")
        _anonymizer(base, anonymize_patient_folders=True).run()
        [row] = _rows(base)
        code = row["patient_code"]
        assert row["anonymized_path"].startswith(f"ANONYMIZED/xml/RAW/{code}/{code} V2/")

    def test_needs_patients_dir_name(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file])
        with pytest.raises(ValueError):
            Anonymizer(base, anonymize_patient_folders=True)

    def test_switching_on_and_off_moves_the_copies(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file], patient="DOE Ann")
        _anonymizer(base).run()
        [before] = _rows(base)
        [r] = _anonymizer(base, anonymize_patient_folders=True).run()
        assert r.ok
        [after] = _rows(base)
        out = base / "dataset-a" / "ANONYMIZED"
        assert after["status"] == before["status"] == "anonymized"
        assert after["patient_code"] == before["patient_code"]
        assert after["ecg_code"] == before["ecg_code"]
        assert not (base / "dataset-a" / before["anonymized_path"]).exists()
        assert not (out / "xml" / "RAW" / "DOE Ann").exists()  # emptied folder removed
        assert len([p for p in out.rglob("*") if p.is_file()]) == 1
        [r] = _anonymizer(base).run()
        [back] = _rows(base)
        assert r.ok and back["anonymized_path"] == before["anonymized_path"]
        assert len([p for p in out.rglob("*") if p.is_file()]) == 1
        [r] = _anonymizer(base).run()
        assert r.unchanged == 1


class TestDateInPatientIdField:
    def _file(self, tmp_path: Path, folder: str, name: str = "R^ECG^F^0^28 11 23^DOE_20231128.xml") -> Path:
        from tests.conftest import HL7_AECG_XML
        path = tmp_path / "export" / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HL7_AECG_XML.replace("SUBJ-001", "28 11 23"), encoding="utf-8")
        return path

    def test_date_kept_in_file_and_name(self, tmp_path):
        self._file(tmp_path, "PATIENT-001")
        [r] = Anonymizer(tmp_path / "export").run()
        assert r.ok
        with open(tmp_path / "export" / "anonymization_catalog.csv", newline="") as f:
            [row] = list(csv.DictReader(f))
        assert row["original_patient_id"] == ""
        assert row["risk"] == "TRUE" and "holds a date (28 11 23)" in row["risk_reason"]
        assert "28 11 23" in row["anonymized_file_name"]
        assert "DOE" not in row["anonymized_file_name"]
        out = (tmp_path / "export" / row["anonymized_path"]).read_text()
        assert 'extension="28 11 23"' in out

    def test_date_does_not_group_patients(self, tmp_path):
        from tests.conftest import HL7_AECG_XML
        self._file(tmp_path, "PATIENT-001")
        other = tmp_path / "export" / "PATIENT-002" / "b.xml"
        other.parent.mkdir(parents=True)
        other.write_text(HL7_AECG_XML.replace("SUBJ-001", "28 11 23").replace("Doe", "Roe"),
                         encoding="utf-8")
        [r] = Anonymizer(tmp_path / "export").run()
        assert r.ok and r.patients == 2 and r.patient_codes == 2


class TestSummaryCounts:
    def test_counts_match(self, hl7_aecg_file, mortara_file, tmp_path, capsys):
        base = _source(tmp_path, [hl7_aecg_file, mortara_file], patient="P1")
        _source(tmp_path, [hl7_aecg_file], patient="P2")
        [r] = _anonymizer(base).run()
        assert (r.raw_files, r.anonymized_files, r.ecg_codes) == (3, 3, 3)
        assert (r.patients, r.patient_codes) == (2, 2)
        assert main(["anonymize", "run", str(base), "--datasets", "--patients-dir-name", "RAW"]) == 0
        out = capsys.readouterr().out
        assert "3 raw file(s) -> 3 anonymized, 3 ECG pseudonym(s); 2 patient(s) -> 2 patient pseudonym(s)" in out


class TestPatientFolderNameAsIdentifier:
    FOLDER = "PAT-0042-XY"

    def _source_with_folder_name_inside(self, tmp_path: Path) -> Path:
        from tests.conftest import HL7_AECG_XML
        text = HL7_AECG_XML.replace("</AnnotatedECG>",
                                    f"<!-- exported for {self.FOLDER} -->\n</AnnotatedECG>")
        raw = tmp_path / "source" / "dataset-a" / "RAW" / self.FOLDER / "V1"
        raw.mkdir(parents=True)
        (raw / f"{self.FOLDER}_20231128.xml").write_text(text, encoding="utf-8")
        return tmp_path / "source"

    def test_folder_name_replaced_in_file_and_name(self, tmp_path):
        base = self._source_with_folder_name_inside(tmp_path)
        [r] = _anonymizer(base, anonymize_patient_folders=True).run()
        assert r.ok
        [row] = _rows(base)
        code = row["patient_code"]
        assert row["anonymized_file_name"] == f"{code}_20231128.xml"
        out = (base / "dataset-a" / row["anonymized_path"]).read_text()
        assert self.FOLDER not in out and f"exported for {code}" in out
        assert self.FOLDER not in row["anonymized_path"]

    def test_folder_name_left_alone_without_the_switch(self, tmp_path):
        base = self._source_with_folder_name_inside(tmp_path)
        [r] = _anonymizer(base).run()
        assert r.ok
        [row] = _rows(base)
        out = (base / "dataset-a" / row["anonymized_path"]).read_text()
        assert f"exported for {self.FOLDER}" in out

    def test_folder_named_like_the_format(self, hl7_aecg_file, tmp_path):
        # The format name is set by the library, not read from the file:
        # a patient folder called "HL7" must not make the check fail
        base = _source(tmp_path, [hl7_aecg_file], patient="HL7")
        [r] = _anonymizer(base, anonymize_patient_folders=True).run()
        assert r.ok and r.failed == 0


class TestGroupingSettingKept:
    def test_adding_patients_dir_name_later_is_refused(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file])
        dataset = base / "dataset-a"
        [r] = Anonymizer(dataset).run()
        assert r.ok
        with pytest.raises(ValueError, match="without --patients-dir-name"):
            Anonymizer(dataset, patients_dir_name="RAW").run()
        assert main(["anonymize", "run", str(dataset), "--patients-dir-name", "RAW"]) == 2
        [r] = Anonymizer(dataset).run()  # the same setting still works
        assert r.ok and r.unchanged == 1

    def test_dropping_patients_dir_name_later_is_refused(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file])
        dataset = base / "dataset-a"
        Anonymizer(dataset, patients_dir_name="RAW").run()
        with pytest.raises(ValueError, match="with --patients-dir-name"):
            Anonymizer(dataset).run()

    def test_starting_again_after_deleting_outputs(self, hl7_aecg_file, tmp_path):
        base = _source(tmp_path, [hl7_aecg_file])
        dataset = base / "dataset-a"
        Anonymizer(dataset).run()
        shutil.rmtree(dataset / "ANONYMIZED")
        (dataset / "anonymization_catalog.csv").unlink()
        [r] = Anonymizer(dataset, patients_dir_name="RAW", anonymize_patient_folders=True).run()
        assert r.ok and r.anonymized == 1
