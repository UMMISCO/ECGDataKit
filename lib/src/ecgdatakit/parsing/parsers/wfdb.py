"""WFDB (PhysioNet/MIT format) parser.

Parses .hea header files + companion .dat binary signal files.
Pure Python implementation supporting Format 16 and Format 212.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError, MissingElementError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    Interpretation,
    Lead,
    PatientInfo,
    RecordingInfo,
)
from ecgdatakit.parsing.parser import Parser

_HEADER_RE = re.compile(
    r"^(\S+)\s+(\d+)\s*(\d+\.?\d*)?\s*"
)


class WFDBParser(Parser):
    """Parser for WFDB (PhysioNet/MIT) ECG files."""

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if file_path.suffix.lower() != ".hea":
            return False
        try:
            text = header.decode("ascii", errors="ignore")
            first_line = text.split("\n")[0].strip()
            return bool(_HEADER_RE.match(first_line))
        except Exception:
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        hea_path = Path(file_path)
        if not hea_path.exists():
            raise FileNotFoundError(f"Header file not found: {hea_path}")

        hea_text = hea_path.read_text(encoding="ascii", errors="replace")
        lines = [l.strip() for l in hea_text.split("\n") if l.strip() and not l.strip().startswith("#")]

        if not lines:
            raise CorruptedFileError("Empty WFDB header file")

        record = ECGRecord(source_format="wfdb")

        first_line = lines[0]
        parts = first_line.split()
        record_name = parts[0]
        num_signals = int(parts[1])

        sample_rate = 250
        if len(parts) >= 3:
            sr_part = parts[2].split("/")[0]
            try:
                sample_rate = int(float(sr_part))
            except ValueError:
                pass

        num_samples = 0
        if len(parts) >= 4:
            try:
                num_samples = int(parts[3])
            except ValueError:
                pass

        base_time = None
        base_date = None
        if len(parts) >= 5:
            try:
                base_time = datetime.strptime(parts[4], "%H:%M:%S")
            except ValueError:
                try:
                    base_time = datetime.strptime(parts[4], "%H:%M:%S.%f")
                except ValueError:
                    pass
        if len(parts) >= 6:
            try:
                base_date = datetime.strptime(parts[5], "%d/%m/%Y")
            except ValueError:
                pass

        signal_specs: list[dict] = []
        for i in range(1, min(num_signals + 1, len(lines))):
            spec = self._parse_signal_line(lines[i])
            signal_specs.append(spec)

        patient = PatientInfo()
        context: dict = {
            "interpretation_statements": [],
            "medications": [],
        }
        for line in hea_text.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                comment = line[1:].strip()
                self._parse_comment(comment, patient, context)

        if context["medications"]:
            patient.medications.extend(context["medications"])

        record.patient = patient

        if context["interpretation_statements"]:
            record.interpretation = Interpretation(
                statements=context["interpretation_statements"],
            )

        recording = RecordingInfo()
        recording.sample_rate = sample_rate

        if base_date and base_time:
            recording.date = base_date.replace(
                hour=base_time.hour,
                minute=base_time.minute,
                second=base_time.second,
            )
        elif base_date:
            recording.date = base_date

        if num_samples > 0 and sample_rate > 0:
            recording.duration = timedelta(seconds=num_samples / sample_rate)

        record.recording = recording

        if signal_specs:
            record.leads = self._read_signals(
                hea_path, signal_specs, num_signals, num_samples, sample_rate
            )

        record.raw_metadata["filepath"] = str(file_path)
        record.raw_metadata["record_name"] = record_name
        record.raw_metadata["signal_specs"] = signal_specs

        return record

    def _parse_signal_line(self, line: str) -> dict:
        """Parse a WFDB signal specification line.

        Format: filename format [samples_per_frame:skew+byte_offset] ADC_gain [(baseline)/ADC_units] ADC_resolution ADC_zero initial_value checksum block_size description
        """
        parts = line.split()
        spec: dict = {
            "filename": parts[0] if len(parts) > 0 else "",
            "format": int(parts[1].split("x")[0]) if len(parts) > 1 else 16,
            "gain": 200.0,
            "baseline": 0,
            "units": "mV",
            "adc_resolution": 12,
            "adc_zero": 0,
            "initial_value": 0,
            "description": "",
        }

        if len(parts) > 2:
            gain_str = parts[2]
            gain_match = re.match(r"([0-9.eE+-]+)(?:\((-?\d+)\))?(?:/(.+))?", gain_str)
            if gain_match:
                try:
                    spec["gain"] = float(gain_match.group(1))
                except ValueError:
                    pass
                if gain_match.group(2):
                    spec["baseline"] = int(gain_match.group(2))
                if gain_match.group(3):
                    spec["units"] = gain_match.group(3)

        if len(parts) > 3:
            try:
                spec["adc_resolution"] = int(parts[3])
            except ValueError:
                pass
        if len(parts) > 4:
            try:
                spec["adc_zero"] = int(parts[4])
            except ValueError:
                pass
        if len(parts) > 5:
            try:
                spec["initial_value"] = int(parts[5])
            except ValueError:
                pass
        if len(parts) > 6:
            try:
                spec["checksum"] = int(parts[6])
            except ValueError:
                spec["checksum"] = 0
        if len(parts) > 7:
            try:
                spec["block_size"] = int(parts[7])
            except ValueError:
                spec["block_size"] = 0
        if len(parts) > 8:
            spec["description"] = " ".join(parts[8:])

        return spec

    def _parse_comment(
        self, comment: str, patient: PatientInfo, context: dict
    ) -> None:
        """Extract patient info and extra metadata from WFDB comment lines.

        Parameters
        ----------
        comment : str
            A single comment line (without the leading ``#``).
        patient : PatientInfo
            Patient object to populate directly.
        context : dict
            Accumulator for values that need post-processing.  Expected
            keys (initialised by the caller):

            * ``"interpretation_statements"`` -- list[str]
            * ``"medications"`` -- list[str]
        """
        if ":" in comment:
            key, _, value = comment.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key in ("age", "patient age"):
                try:
                    patient.age = int(value)
                except ValueError:
                    pass
            elif key in ("sex", "gender"):
                v = value.upper()
                patient.sex = "M" if v in ("M", "MALE") else "F" if v in ("F", "FEMALE") else "U"
            elif key in ("id", "patient id"):
                patient.patient_id = value
            elif key in ("height", "patient height"):
                try:
                    patient.height = float(value)
                except ValueError:
                    pass
            elif key in ("weight", "patient weight"):
                try:
                    patient.weight = float(value)
                except ValueError:
                    pass
            elif key in ("dx", "diagnosis", "diagnoses"):
                if value:
                    context["interpretation_statements"].append(value)
            elif key in ("drugs", "medications", "medication"):
                if value:
                    context["medications"].append(value)
            elif key in ("name", "patient name"):
                if value:
                    name_parts = value.split(None, 1)
                    patient.first_name = name_parts[0]
                    if len(name_parts) > 1:
                        patient.last_name = name_parts[1]

    def _read_signals(
        self,
        hea_path: Path,
        signal_specs: list[dict],
        num_signals: int,
        num_samples: int,
        sample_rate: int,
    ) -> list[Lead]:
        """Read signal data from .dat file."""
        leads: list[Lead] = []

        files: dict[str, list[int]] = {}
        for i, spec in enumerate(signal_specs):
            fname = spec["filename"]
            if fname not in files:
                files[fname] = []
            files[fname].append(i)

        for dat_filename, sig_indices in files.items():
            dat_path = hea_path.parent / dat_filename
            if not dat_path.exists():
                continue

            dat_bytes = dat_path.read_bytes()
            fmt = signal_specs[sig_indices[0]]["format"]
            n_channels = len(sig_indices)

            if fmt == 16:
                raw = np.frombuffer(dat_bytes, dtype="<i2")
                if n_channels > 0:
                    samples_per_channel = len(raw) // n_channels
                    raw = raw[:samples_per_channel * n_channels]
                    channels = raw.reshape((samples_per_channel, n_channels))

                    for ch_offset, sig_idx in enumerate(sig_indices):
                        spec = signal_specs[sig_idx]
                        gain = spec["gain"] if spec["gain"] != 0 else 200.0
                        baseline = spec["baseline"]
                        samples = (channels[:, ch_offset].astype(np.float64) - baseline) / gain
                        label = spec["description"] or f"Ch{sig_idx}"

                        leads.append(Lead(
                            label=label,
                            samples=samples,
                            sample_rate=sample_rate,
                            units=spec["units"],
                        ))

            elif fmt == 212:
                all_samples = self._decode_format_212(dat_bytes, n_channels, num_samples)

                for ch_offset, sig_idx in enumerate(sig_indices):
                    spec = signal_specs[sig_idx]
                    gain = spec["gain"] if spec["gain"] != 0 else 200.0
                    baseline = spec["baseline"]

                    if ch_offset < all_samples.shape[1]:
                        samples = (all_samples[:, ch_offset].astype(np.float64) - baseline) / gain
                    else:
                        samples = np.array([], dtype=np.float64)

                    label = spec["description"] or f"Ch{sig_idx}"
                    leads.append(Lead(
                        label=label,
                        samples=samples,
                        sample_rate=sample_rate,
                        units=spec["units"],
                    ))

            else:
                continue

        return leads

    @staticmethod
    def _decode_format_212(data: bytes, n_channels: int, num_samples: int) -> np.ndarray:
        """Decode Format 212 (12-bit packed) signal data."""
        raw = np.frombuffer(data, dtype=np.uint8)
        n_pairs = len(raw) // 3
        if n_pairs == 0:
            return np.array([], dtype=np.int16).reshape((0, max(n_channels, 1)))

        raw_trimmed = raw[:n_pairs * 3].reshape((n_pairs, 3))

        s0 = raw_trimmed[:, 0].astype(np.int16) | ((raw_trimmed[:, 1].astype(np.int16) & 0x0F) << 8)
        s1 = (raw_trimmed[:, 1].astype(np.int16) >> 4) | (raw_trimmed[:, 2].astype(np.int16) << 4)

        s0 = np.where(s0 >= 2048, s0 - 4096, s0)
        s1 = np.where(s1 >= 2048, s1 - 4096, s1)

        all_samples = np.empty(n_pairs * 2, dtype=np.int16)
        all_samples[0::2] = s0
        all_samples[1::2] = s1

        total = len(all_samples)
        samples_per_channel = total // n_channels
        all_samples = all_samples[:samples_per_channel * n_channels]

        return all_samples.reshape((samples_per_channel, n_channels))
