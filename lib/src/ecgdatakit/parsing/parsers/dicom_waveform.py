"""DICOM Waveform ECG parser.

Parses DICOM Part 10 files containing ECG waveform data.
Requires pydicom (optional dependency).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError, MissingElementError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    GlobalMeasurements,
    Interpretation,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
)
from ecgdatakit.parsing.parser import Parser


def _import_pydicom():
    """Lazily import pydicom, raising a helpful error if missing."""
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM waveform parsing. "
            'Install it with: pip install "ecgdatakit[dicom]"'
        )


class DICOMWaveformParser(Parser):
    """Parser for DICOM Waveform ECG files."""

    FORMAT_NAME = "DICOM Waveform"
    FORMAT_DESCRIPTION = "DICOM Waveform ECG (Part 10, requires pydicom)"
    FILE_EXTENSIONS = [".dcm", ".dicom"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if len(header) < 132:
            return False
        return header[128:132] == b"DICM"

    def parse(self, file_path: Path) -> ECGRecord:
        pydicom = _import_pydicom()

        try:
            ds = pydicom.dcmread(str(file_path))
        except Exception as e:
            raise CorruptedFileError(f"Failed to read DICOM file: {e}")

        record = ECGRecord(source_format="dicom_waveform")

        record.patient = self._read_patient(ds)
        record.recording = self._read_recording(ds)
        record.recording.device = self._read_device(ds)
        leads, filters, signal = self._read_leads(ds)
        record.leads = leads
        record.recording.acquisition.filters = filters
        record.recording.acquisition.signal = signal
        if record.leads:
            if record.recording.acquisition.signal.sample_rate == 0:
                record.recording.acquisition.signal.sample_rate = record.leads[0].sample_rate
            if record.recording.acquisition.signal.sample_rate > 0 and len(record.leads[0].samples) > 0:
                duration_s = len(record.leads[0].samples) / record.recording.acquisition.signal.sample_rate
                record.recording.duration = timedelta(seconds=duration_s)

        record.interpretation, record.measurements = self._read_annotations(ds)

        # Extract technician and department
        technician = str(getattr(ds, "OperatorsName", ""))
        if technician:
            record.recording.technician = technician
        department = str(getattr(ds, "InstitutionalDepartmentName", ""))
        if department:
            record.recording.device.department = department

        record.raw_metadata["filepath"] = str(file_path)
        record.raw_metadata["sop_class_uid"] = str(getattr(ds, "SOPClassUID", ""))
        record.raw_metadata["modality"] = str(getattr(ds, "Modality", ""))

        study_desc = str(getattr(ds, "StudyDescription", ""))
        if study_desc:
            record.raw_metadata["study_description"] = study_desc

        return record

    def _read_patient(self, ds) -> PatientInfo:
        info = PatientInfo()

        info.patient_id = str(getattr(ds, "PatientID", ""))

        name = getattr(ds, "PatientName", None)
        if name is not None:
            name_str = str(name)
            parts = name_str.split("^")
            if len(parts) >= 1:
                info.last_name = parts[0].strip()
            if len(parts) >= 2:
                info.first_name = parts[1].strip()

        sex = str(getattr(ds, "PatientSex", ""))
        info.sex = "M" if sex == "M" else "F" if sex == "F" else "U"

        dob = str(getattr(ds, "PatientBirthDate", ""))
        if dob and len(dob) >= 8:
            try:
                info.birth_date = datetime.strptime(dob[:8], "%Y%m%d")
            except ValueError:
                pass

        age_str = str(getattr(ds, "PatientAge", ""))
        if age_str:
            age_str = age_str.replace("Y", "").replace("y", "").strip()
            if age_str.isdigit():
                info.age = int(age_str)

        weight = getattr(ds, "PatientWeight", None)
        if weight is not None:
            try:
                info.weight = float(weight)
            except (ValueError, TypeError):
                pass

        height = getattr(ds, "PatientSize", None)
        if height is not None:
            try:
                info.height = float(height) * 100
            except (ValueError, TypeError):
                pass

        return info

    def _read_recording(self, ds) -> RecordingInfo:
        info = RecordingInfo()

        date_str = str(getattr(ds, "AcquisitionDate", "")) or str(getattr(ds, "StudyDate", ""))
        time_str = str(getattr(ds, "AcquisitionTime", "")) or str(getattr(ds, "StudyTime", ""))

        if date_str and len(date_str) >= 8:
            try:
                dt = datetime.strptime(date_str[:8], "%Y%m%d")
                if time_str and len(time_str) >= 6:
                    t = datetime.strptime(time_str[:6], "%H%M%S")
                    dt = dt.replace(hour=t.hour, minute=t.minute, second=t.second)
                info.date = dt
            except ValueError:
                pass

        info.referring_physician = str(getattr(ds, "ReferringPhysicianName", ""))

        return info

    def _read_device(self, ds) -> DeviceInfo:
        info = DeviceInfo()

        info.manufacturer = str(getattr(ds, "Manufacturer", ""))
        info.model = str(getattr(ds, "ManufacturerModelName", ""))
        info.software_version = str(getattr(ds, "SoftwareVersions", ""))
        info.institution = str(getattr(ds, "InstitutionName", ""))
        info.serial_number = str(getattr(ds, "DeviceSerialNumber", ""))

        return info

    def _read_leads(self, ds) -> tuple[list[Lead], FilterSettings, SignalCharacteristics]:
        leads: list[Lead] = []
        filters = FilterSettings()
        sig = SignalCharacteristics(
            data_encoding="pcm",
            compression="none",
        )
        highpass_values: list[float] = []
        lowpass_values: list[float] = []
        notch_values: list[float] = []

        waveform_seq = getattr(ds, "WaveformSequence", None)
        if waveform_seq is None:
            return leads, filters, sig

        for wf in waveform_seq:
            num_channels = int(getattr(wf, "NumberOfWaveformChannels", 0))
            num_samples = int(getattr(wf, "NumberOfWaveformSamples", 0))
            sample_rate = float(getattr(wf, "SamplingFrequency", 0))
            bits_allocated = int(getattr(wf, "WaveformBitsAllocated", 16))

            sig.bits_per_sample = bits_allocated
            sig.number_channels_allocated = num_channels

            # Determine signedness from WaveformSampleInterpretation
            sample_interp = str(getattr(wf, "WaveformSampleInterpretation", ""))
            if sample_interp in ("SB", "SS"):
                sig.signal_signed = True
            elif sample_interp in ("UB", "US"):
                sig.signal_signed = False

            channel_defs = getattr(wf, "ChannelDefinitionSequence", [])
            waveform_data = getattr(wf, "WaveformData", None)

            if waveform_data is None or num_channels == 0:
                continue

            if bits_allocated == 16:
                dtype = np.int16
            elif bits_allocated == 32:
                dtype = np.int32
            else:
                dtype = np.int16

            raw_data = np.frombuffer(waveform_data, dtype=dtype)

            if len(raw_data) < num_channels * num_samples:
                num_samples = len(raw_data) // num_channels

            if num_samples == 0:
                continue

            raw_data = raw_data[:num_channels * num_samples]
            channels = raw_data.reshape((num_samples, num_channels))

            for ch_idx in range(num_channels):
                label = f"Ch{ch_idx}"
                sensitivity = 1.0
                baseline = 0.0
                units = ""

                if ch_idx < len(channel_defs):
                    ch_def = channel_defs[ch_idx]

                    source_seq = getattr(ch_def, "ChannelSourceSequence", None)
                    if source_seq and len(source_seq) > 0:
                        code_meaning = str(getattr(source_seq[0], "CodeMeaning", ""))
                        if code_meaning:
                            label = code_meaning
                            label = label.replace("Lead ", "").replace("lead ", "")

                    sens = getattr(ch_def, "ChannelSensitivity", None)
                    if sens is not None:
                        try:
                            sensitivity = float(sens)
                        except (ValueError, TypeError):
                            pass

                    bl = getattr(ch_def, "ChannelBaseline", None)
                    if bl is not None:
                        try:
                            baseline = float(bl)
                        except (ValueError, TypeError):
                            pass

                    correction = getattr(ch_def, "ChannelSensitivityCorrectionFactor", None)
                    if correction is not None:
                        try:
                            sensitivity *= float(correction)
                        except (ValueError, TypeError):
                            pass

                    units_seq = getattr(ch_def, "ChannelSensitivityUnitsSequence", None)
                    if units_seq and len(units_seq) > 0:
                        units = str(getattr(units_seq[0], "CodeMeaning", ""))

                    low_freq = getattr(ch_def, "FilterLowFrequency", None)
                    if low_freq is not None:
                        try:
                            highpass_values.append(float(low_freq))
                        except (ValueError, TypeError):
                            pass

                    high_freq = getattr(ch_def, "FilterHighFrequency", None)
                    if high_freq is not None:
                        try:
                            lowpass_values.append(float(high_freq))
                        except (ValueError, TypeError):
                            pass

                    notch_freq = getattr(ch_def, "NotchFilterFrequency", None)
                    if notch_freq is not None:
                        try:
                            notch_values.append(float(notch_freq))
                        except (ValueError, TypeError):
                            pass

                samples = channels[:, ch_idx].astype(np.float64)
                offset_val = -baseline * sensitivity
                raw = not (sensitivity == 1.0 and offset_val == 0.0)
                leads.append(Lead(
                    label=label,
                    samples=samples,
                    sample_rate=int(sample_rate),
                    resolution=sensitivity,
                    resolution_unit=units,
                    offset=offset_val,
                    units="" if raw else units,
                    is_raw=raw,
                ))

            if leads:
                break

        if highpass_values:
            filters.highpass = highpass_values[0]
        if lowpass_values:
            filters.lowpass = lowpass_values[0]
        if notch_values:
            filters.notch = notch_values[0]
            filters.notch_active = True

        sig.number_channels_valid = len(leads)

        return leads, filters, sig

    _MEASUREMENT_CODES: dict[str, str] = {
        "MDC_ECG_HEART_RATE": "heart_rate",
        "MDC_ECG_TIME_PD_RR": "rr_interval",
        "MDC_ECG_TIME_PD_PR": "pr_interval",
        "MDC_ECG_TIME_PD_QRS": "qrs_duration",
        "MDC_ECG_TIME_PD_QT": "qt_interval",
        "MDC_ECG_TIME_PD_QTc": "qtc_bazett",
        "MDC_ECG_TIME_PD_QTcB": "qtc_bazett",
        "MDC_ECG_TIME_PD_QTcF": "qtc_fridericia",
        "MDC_ECG_ANGLE_P_FRONT": "p_axis",
        "MDC_ECG_ANGLE_QRS_FRONT": "qrs_axis",
        "MDC_ECG_ANGLE_T_FRONT": "t_axis",
        "5.10.2.1-1": "heart_rate",
        "5.10.3-1": "rr_interval",
        "5.10.3-2": "pr_interval",
        "5.10.3-3": "qrs_duration",
        "5.10.3-4": "qt_interval",
        "5.10.3-5": "qtc_bazett",
    }

    def _read_annotations(
        self, ds,
    ) -> tuple[Interpretation, GlobalMeasurements]:
        """Parse WaveformAnnotationSequence for interpretation and measurements."""
        interp = Interpretation()
        meas = GlobalMeasurements()

        annotation_items: list = []

        top_ann = getattr(ds, "WaveformAnnotationSequence", None)
        if top_ann is not None:
            annotation_items.extend(top_ann)

        waveform_seq = getattr(ds, "WaveformSequence", None)
        if waveform_seq is not None:
            for wf in waveform_seq:
                wf_ann = getattr(wf, "WaveformAnnotationSequence", None)
                if wf_ann is not None:
                    annotation_items.extend(wf_ann)

        if not annotation_items:
            return interp, meas

        statements: list[tuple[str, str]] = []
        measurement_map: dict[str, float] = {}

        for ann in annotation_items:
            concept_seq = getattr(ann, "ConceptNameCodeSequence", None)
            code_value = ""
            code_meaning = ""
            if concept_seq and len(concept_seq) > 0:
                code_value = str(getattr(concept_seq[0], "CodeValue", ""))
                code_meaning = str(getattr(concept_seq[0], "CodeMeaning", ""))

            numeric_value = getattr(ann, "NumericValue", None)
            if numeric_value is not None:
                try:
                    val = float(numeric_value)
                except (ValueError, TypeError):
                    val = None

                if val is not None:
                    field_name = self._MEASUREMENT_CODES.get(code_value)
                    if field_name is None:
                        field_name = self._MEASUREMENT_CODES.get(code_meaning)
                    if field_name is not None:
                        measurement_map[field_name] = val

            text = str(getattr(ann, "UnformattedTextValue", "")).strip()
            if text:
                statements.append((text, ""))
            elif code_meaning and numeric_value is None:
                statements.append((code_meaning, ""))

        if statements:
            interp.statements = statements
            interp.source = "machine"

        if "heart_rate" in measurement_map:
            meas.heart_rate = int(measurement_map["heart_rate"])
        if "rr_interval" in measurement_map:
            meas.rr_interval = int(measurement_map["rr_interval"])
        if "pr_interval" in measurement_map:
            meas.pr_interval = int(measurement_map["pr_interval"])
        if "qrs_duration" in measurement_map:
            meas.qrs_duration = int(measurement_map["qrs_duration"])
        if "qt_interval" in measurement_map:
            meas.qt_interval = int(measurement_map["qt_interval"])
        if "qtc_bazett" in measurement_map:
            meas.qtc_bazett = int(measurement_map["qtc_bazett"])
        if "qtc_fridericia" in measurement_map:
            meas.qtc_fridericia = int(measurement_map["qtc_fridericia"])
        if "p_axis" in measurement_map:
            meas.p_axis = int(measurement_map["p_axis"])
        if "qrs_axis" in measurement_map:
            meas.qrs_axis = int(measurement_map["qrs_axis"])
        if "t_axis" in measurement_map:
            meas.t_axis = int(measurement_map["t_axis"])

        return interp, meas
