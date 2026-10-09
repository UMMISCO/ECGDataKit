"""DICOM waveform anonymization.

Elements replaced: PatientName, PatientID, OtherPatientIDs (and
OtherPatientIDsSequence), OtherPatientNames and PatientBirthName become the
patient code. SOPInstanceUID (and MediaStorageSOPInstanceUID in the file
meta information) becomes a UID derived from the ECG code (2.25 root).
Other text elements (LO, SH, ST, LT, UT, PN) are searched for names.

The file is written back with pydicom, keeping its transfer syntax and
every other element. The waveform data is not touched.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from ecgdatakit.anonymize._codes import code_dicom_uid
from ecgdatakit.anonymize._identity import Identity, Replacer
from ecgdatakit.anonymize.formats._base import Codes, Handler, Rewrite

_DICOM_TEXT_VRS = {"LO", "SH", "ST", "LT", "UT", "PN"}
_DICOM_PATIENT_TAGS = ("PatientName", "PatientID", "OtherPatientIDs", "OtherPatientNames",
                       "PatientBirthName")


def _pydicom():
    try:
        import pydicom
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("DICOM anonymization needs pydicom: pip install 'ecgdatakit[dicom]'") from e
    return pydicom


class DICOMHandler(Handler):
    """DICOM waveform: patient name and IDs, and the SOP Instance UID.

    The file is written back with pydicom, keeping its transfer syntax and
    every other element. The waveform data is not touched.
    """

    parser = "DICOMWaveformParser"

    def _read(self, path):
        return _pydicom().dcmread(str(path), force=True)

    def identity(self, path: Path) -> Identity:
        ds = self._read(path)
        identity = Identity()
        name = ds.get("PatientName")
        if name is not None and str(name):
            identity.add("last_names", name.family_name or str(name))
            identity.add("first_names", name.given_name)
            identity.add("first_names", name.middle_name)
        identity.add("patient_ids", str(ds.get("PatientID", "") or ""))
        others = ds.get("OtherPatientIDs")
        for value in (others if isinstance(others, (list, tuple)) or type(others).__name__ == "MultiValue"
                      else [others] if others else []):
            identity.add("patient_ids", str(value))
        for item in ds.get("OtherPatientIDsSequence", []) or []:
            identity.add("patient_ids", str(item.get("PatientID", "") or ""))
        for tag in ("OtherPatientNames", "PatientBirthName"):
            value = ds.get(tag)
            if value:
                for v in (value if type(value).__name__ == "MultiValue" else [value]):
                    identity.add("last_names", str(v).replace("^", " ").strip())
        identity.add("ecg_ids", str(ds.get("SOPInstanceUID", "") or ""))
        return identity

    def rewrite(self, path, out_path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        ds = self._read(path)
        result = Rewrite()
        for keyword in _DICOM_PATIENT_TAGS:
            if keyword in ds and ds.data_element(keyword).value:
                ds.data_element(keyword).value = codes.patient
        for item in ds.get("OtherPatientIDsSequence", []) or []:
            if item.get("PatientID"):
                item.PatientID = codes.patient
        old_uid = str(ds.get("SOPInstanceUID", "") or "")
        if old_uid:
            new_uid = code_dicom_uid(codes.ecg)
            ds.SOPInstanceUID = new_uid
            meta = getattr(ds, "file_meta", None)
            if meta is not None and str(meta.get("MediaStorageSOPInstanceUID", "")) == old_uid:
                meta.MediaStorageSOPInstanceUID = new_uid
            result.ecg_written.append(new_uid)
        if replacer:
            def walk(dataset):
                for element in dataset:
                    if element.VR == "SQ":
                        for item in element.value:
                            walk(item)
                    elif element.VR in _DICOM_TEXT_VRS and element.keyword not in _DICOM_PATIENT_TAGS:
                        value = element.value
                        if isinstance(value, str):
                            new = replacer.replace(value)
                            if new != value:
                                element.value = new
                        elif type(value).__name__ == "MultiValue":
                            new = [replacer.replace(str(v)) for v in value]
                            if new != [str(v) for v in value]:
                                element.value = new
                        elif value is not None and element.VR == "PN":
                            new = replacer.replace(str(value))
                            if new != str(value):
                                element.value = new
            walk(ds)
        buffer = BytesIO()
        try:
            ds.save_as(buffer, enforce_file_format=False)
        except TypeError:  # pydicom < 3
            ds.save_as(buffer, write_like_original=True)
        result.outputs[Path(out_path)] = buffer.getvalue()
        return result

    def free_text(self, data: bytes) -> list[str]:
        ds = _pydicom().dcmread(BytesIO(data), force=True, stop_before_pixels=True)
        parts: list[str] = []

        def walk(dataset):
            for element in dataset:
                if element.VR == "SQ":
                    for item in element.value:
                        walk(item)
                elif element.VR in _DICOM_TEXT_VRS | {"UI"} and element.value is not None:
                    parts.append(str(element.value))
        walk(ds)
        return parts
