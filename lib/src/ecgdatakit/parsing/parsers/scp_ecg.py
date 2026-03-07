"""SCP-ECG (Standard Communication Protocol for ECG) parser.

Reference: ISO/EN 13730 — section-based TLV binary format.
"""

from __future__ import annotations

import struct
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

_DEFAULT_HUFFMAN_TABLE = [
    (1, 0b0, 0),
    (3, 0b100, 1),
    (3, 0b101, -1),
    (4, 0b1100, 2),
    (4, 0b1101, -2),
    (5, 0b11100, 3),
    (5, 0b11101, -3),
    (6, 0b111100, 4),
    (6, 0b111101, -4),
    (7, 0b1111100, 5),
    (7, 0b1111101, -5),
    (8, 0b11111100, 6),
    (8, 0b11111101, -6),
    (9, 0b111111100, 7),
    (9, 0b111111101, -7),
    (10, 0b1111111100, 8),
    (10, 0b1111111101, -8),
    (18, 0b111111111000000000, None),
]

_SCP_LEAD_LABELS = {
    0: "I", 1: "II", 2: "V1", 3: "V2", 4: "V3",
    5: "V4", 6: "V5", 7: "V6", 8: "III", 9: "aVR",
    10: "aVL", 11: "aVF",
}


def _read_u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _read_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _read_i16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<h", data, offset)[0]


def _read_i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _decode_huffman(data: bytes, num_samples: int, use_default: bool = True) -> list[int]:
    """Decode Huffman-encoded SCP-ECG data using default table."""
    if not use_default:
        values = []
        for i in range(0, min(len(data), num_samples * 2), 2):
            values.append(struct.unpack_from("<h", data, i)[0])
        return values[:num_samples]

    values: list[int] = []
    bit_pos = 0
    total_bits = len(data) * 8

    while len(values) < num_samples and bit_pos < total_bits:
        matched = False
        for bits, prefix, value in _DEFAULT_HUFFMAN_TABLE:
            if bit_pos + bits > total_bits:
                continue
            extracted = 0
            for b in range(bits):
                byte_idx = (bit_pos + b) // 8
                bit_idx = 7 - ((bit_pos + b) % 8)
                extracted = (extracted << 1) | ((data[byte_idx] >> bit_idx) & 1)

            if extracted == prefix:
                if value is None:
                    bit_pos += bits
                    if bit_pos + 16 > total_bits:
                        break
                    raw = 0
                    for b in range(16):
                        byte_idx = (bit_pos + b) // 8
                        bit_idx = 7 - ((bit_pos + b) % 8)
                        raw = (raw << 1) | ((data[byte_idx] >> bit_idx) & 1)
                    if raw >= 0x8000:
                        raw -= 0x10000
                    values.append(raw)
                    bit_pos += 16
                else:
                    values.append(value)
                    bit_pos += bits
                matched = True
                break

        if not matched:
            bit_pos += 1

    return values[:num_samples]


def _reconstruct_first_difference(diffs: list[int]) -> list[int]:
    """Reconstruct signal from first-difference encoding."""
    if not diffs:
        return []
    result = [diffs[0]]
    for i in range(1, len(diffs)):
        result.append(result[-1] + diffs[i])
    return result


def _reconstruct_second_difference(diffs: list[int]) -> list[int]:
    """Reconstruct signal from second-difference encoding."""
    if len(diffs) < 2:
        return diffs
    result = [diffs[0], diffs[1]]
    for i in range(2, len(diffs)):
        result.append(2 * result[-1] - result[-2] + diffs[i])
    return result


class SCPECGParser(Parser):
    """Parser for SCP-ECG binary ECG files (ISO 13730)."""

    FORMAT_NAME = "SCP-ECG"
    FORMAT_DESCRIPTION = "Standard Communications Protocol for ECG (ISO 13730)"
    FILE_EXTENSIONS = [".scp", ".ecg"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if len(header) < 16:
            return False
        try:
            file_size = struct.unpack_from("<I", header, 2)[0]
            if file_size < 16 or file_size > 50_000_000:
                return False

            if len(header) < 8:
                return False

            section_id = header[6]
            if section_id != 0:
                return False

            if len(header) >= 14:
                section_len = struct.unpack_from("<I", header, 8)[0]
                if section_len < 8:
                    return False

            return True
        except Exception:
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            data = f.read()

        if len(data) < 16:
            raise CorruptedFileError(f"File too small for SCP-ECG: {len(data)} bytes")

        record = ECGRecord(source_format="scp_ecg")

        crc = _read_u16(data, 0)
        file_size = _read_u32(data, 2)

        sections: dict[int, tuple[int, int]] = {}
        self._parse_section0(data, sections)

        sec1_ctx: dict = {}
        if 1 in sections:
            offset, length = sections[1]
            sec1_ctx = self._parse_section1(data, offset, length)
            record.patient = sec1_ctx["patient"]
            record.interpretation = sec1_ctx.get("interpretation", Interpretation())
            for key, val in sec1_ctx.get("raw_metadata", {}).items():
                record.raw_metadata[key] = val

        lead_defs: list[dict] = []
        num_leads = 0
        if 3 in sections:
            offset, length = sections[3]
            num_leads, lead_defs = self._parse_section3(data, offset, length)

        use_default_huffman = True
        if 2 in sections:
            offset, length = sections[2]
            if length > 16:
                sec_data_offset = offset + 16
                if sec_data_offset + 2 <= len(data):
                    num_tables = _read_u16(data, sec_data_offset)
                    use_default_huffman = (num_tables == 19999)

        if 4 in sections:
            offset, length = sections[4]
            record.raw_metadata["qrs_locations"] = self._parse_section4(
                data, offset, length
            )

        if 5 in sections:
            offset, length = sections[5]
            record.median_beats = self._parse_section5(
                data, offset, length, lead_defs, use_default_huffman
            )

        if 6 in sections:
            offset, length = sections[6]
            record.leads = self._parse_section6(
                data, offset, length, lead_defs, use_default_huffman
            )

        if 7 in sections:
            offset, length = sections[7]
            record.measurements = self._parse_section7(data, offset, length)

        if 8 in sections:
            offset, length = sections[8]
            sec8_interp = self._parse_section8(data, offset, length)
            if sec8_interp.statements:
                existing = record.interpretation.statements or []
                merged = existing + [
                    s for s in sec8_interp.statements if s not in existing
                ]
                record.interpretation.statements = merged
            if sec8_interp.source and not record.interpretation.source:
                record.interpretation.source = sec8_interp.source

        recording = RecordingInfo()
        sr = 0
        if record.leads:
            sr = record.leads[0].sampling_rate
            if sr > 0:
                duration_s = len(record.leads[0].samples) / sr
                recording.duration = timedelta(seconds=duration_s)

        rec_date = sec1_ctx.get("recording_date")
        if rec_date is not None:
            recording.date = rec_date
        sec1_extra = sec1_ctx.get("raw_metadata", {})
        if "referring_physician" in sec1_extra:
            recording.referring_physician = sec1_extra["referring_physician"]
        if "technician" in sec1_extra:
            recording.technician = sec1_extra["technician"]
        if "room" in sec1_extra:
            recording.room = sec1_extra["room"]

        record.recording = recording
        record.recording.device = sec1_ctx.get("device", DeviceInfo())
        record.recording.acquisition.filters = sec1_ctx.get("filters", FilterSettings())

        # Populate signal characteristics
        sig = SignalCharacteristics(
            sampling_rate=sr,
            signal_signed=True,
            number_channels_allocated=num_leads,
            number_channels_valid=len(record.leads),
            compression="none",
            data_encoding="raw",
        )
        if 6 in sections:
            sec6_offset = sections[6][0] + 16
            if sec6_offset + 6 <= len(data):
                enc_flag = data[sec6_offset + 4]
                comp_flag = data[sec6_offset + 5]
                if comp_flag == 0:
                    sig.compression = "huffman"
                if enc_flag == 1:
                    sig.data_encoding = "second_difference"
                elif enc_flag == 0:
                    sig.data_encoding = "first_difference"
        record.recording.acquisition.signal = sig

        record.raw_metadata["filepath"] = str(file_path)
        record.raw_metadata["crc"] = crc
        record.raw_metadata["file_size"] = file_size
        record.raw_metadata["sections_found"] = list(sections.keys())

        return record

    def _parse_section0(self, data: bytes, sections: dict[int, tuple[int, int]]) -> None:
        """Parse Section 0 pointer table."""
        if len(data) < 14:
            raise CorruptedFileError("File too small for Section 0")

        sec0_offset = 6
        sec0_id = data[sec0_offset]
        if sec0_id != 0:
            raise CorruptedFileError(f"Expected Section 0 at offset 6, got section {sec0_id}")

        sec0_len = _read_u32(data, sec0_offset + 4)
        header_size = 16

        ptr_offset = sec0_offset + header_size
        ptr_end = sec0_offset + sec0_len

        while ptr_offset + 10 <= ptr_end and ptr_offset + 10 <= len(data):
            section_id = _read_u16(data, ptr_offset)
            section_length = _read_u32(data, ptr_offset + 2)
            section_index = _read_u32(data, ptr_offset + 6)

            if section_length > 0 and section_index > 0:
                sections[section_id] = (section_index - 1, section_length)

            ptr_offset += 10

    def _parse_section1(self, data: bytes, offset: int, length: int) -> dict:
        """Parse Section 1: Patient demographics and recording context.

        Returns a dict with keys: ``patient``, ``recording_date``,
        ``device``, ``filters``, ``interpretation``, ``raw_metadata``.
        """
        info = PatientInfo()
        recording_date: datetime | None = None
        recording_time: tuple[int, int, int] | None = None
        device = DeviceInfo()
        filters = FilterSettings()
        interpretation = Interpretation()
        extra: dict = {}

        header_size = 16
        pos = offset + header_size
        end = offset + length

        while pos + 3 <= end and pos + 3 <= len(data):
            tag = data[pos]
            tag_len = _read_u16(data, pos + 1)
            pos += 3

            if pos + tag_len > len(data):
                break

            tag_data = data[pos: pos + tag_len]

            if tag == 0 and tag_len >= 1:
                info.last_name = tag_data.decode("ascii", errors="replace").strip("\x00 ")
            elif tag == 1 and tag_len >= 1:
                info.first_name = tag_data.decode("ascii", errors="replace").strip("\x00 ")
            elif tag == 2 and tag_len >= 1:
                info.patient_id = tag_data.decode("ascii", errors="replace").strip("\x00 ")
            elif tag == 3 and tag_len >= 1:
                extra["second_last_name"] = tag_data.decode("ascii", errors="replace").strip("\x00 ")
            elif tag == 5 and tag_len >= 4:
                try:
                    year = _read_u16(tag_data, 0)
                    month = tag_data[2]
                    day = tag_data[3]
                    if 1 <= month <= 12 and 1 <= day <= 31 and year > 0:
                        recording_date = datetime(year, month, day)
                except (ValueError, IndexError):
                    pass
            elif tag == 6 and tag_len >= 3:
                try:
                    hour = tag_data[0]
                    minute = tag_data[1]
                    second = tag_data[2]
                    recording_time = (hour, minute, second)
                except (ValueError, IndexError):
                    pass
            elif tag == 7 and tag_len >= 2:
                height_val = _read_u16(tag_data, 0)
                if 0 < height_val < 300:
                    info.height = float(height_val)
            elif tag == 8 and tag_len >= 4:
                try:
                    year = _read_u16(tag_data, 0)
                    month = tag_data[2]
                    day = tag_data[3]
                    if 1 <= month <= 12 and 1 <= day <= 31 and year > 0:
                        info.birth_date = datetime(year, month, day)
                except (ValueError, IndexError):
                    pass
            elif tag == 9 and tag_len >= 1:
                sex_code = tag_data[0]
                info.sex = "M" if sex_code == 1 else "F" if sex_code == 2 else "U"
            elif tag == 10 and tag_len >= 1:
                raw = tag_data.decode("ascii", errors="replace").strip("\x00 ")
                meds = [m.strip() for m in raw.split("\x00") if m.strip()]
                if meds:
                    info.medications = meds
            elif tag == 11 and tag_len >= 2:
                age_val = _read_u16(tag_data, 0)
                if 0 < age_val < 200:
                    info.age = age_val
            elif tag == 12 and tag_len >= 2:
                weight_val = _read_u16(tag_data, 0)
                if 0 < weight_val < 500:
                    info.weight = float(weight_val)
            elif tag == 13 and tag_len >= 1:
                extra["referring_physician"] = tag_data.decode("ascii", errors="replace").strip("\x00 ")
            elif tag == 14 and tag_len >= 1:
                interpretation.interpreter = tag_data.decode("ascii", errors="replace").strip("\x00 ")
            elif tag == 15 and tag_len >= 1:
                extra["technician"] = tag_data.decode("ascii", errors="replace").strip("\x00 ")
            elif tag == 16 and tag_len >= 1:
                extra["room"] = tag_data.decode("ascii", errors="replace").strip("\x00 ")
            elif tag == 20 and tag_len >= 1:
                parts = tag_data.decode("ascii", errors="replace").strip("\x00 ").split("\x00")
                if len(parts) >= 1:
                    device.institution = parts[0].strip()
                if len(parts) >= 2:
                    device.department = parts[1].strip()
                if len(parts) >= 3:
                    device.serial_number = parts[2].strip()
            elif tag == 21 and tag_len >= 1:
                parts = tag_data.decode("ascii", errors="replace").strip("\x00 ").split("\x00")
                extra["analyzing_device"] = {
                    "institution": parts[0].strip() if len(parts) >= 1 else "",
                    "department": parts[1].strip() if len(parts) >= 2 else "",
                    "device_id": parts[2].strip() if len(parts) >= 3 else "",
                }
            elif tag == 25 and tag_len >= 1:
                info.clinical_history = tag_data.decode("ascii", errors="replace").strip("\x00 ")
            elif tag == 30 and tag_len >= 1:
                try:
                    if tag_len >= 2:
                        filters.lowpass = float(_read_u16(tag_data, 0))
                    if tag_len >= 4:
                        hipass_raw = _read_u16(tag_data, 2)
                        filters.highpass = hipass_raw / 100.0
                    if tag_len >= 5:
                        notch_val = tag_data[4]
                        if notch_val in (0, 1, 2):
                            filters.notch = {0: None, 1: 60.0, 2: 50.0}.get(notch_val)
                            filters.notch_active = notch_val != 0
                except (IndexError, struct.error):
                    pass
            elif tag == 35 and tag_len >= 1:
                text = tag_data.decode("ascii", errors="replace").strip("\x00 ")
                if text:
                    interpretation.statements = [
                        (s.strip(), "") for s in text.split("\x00") if s.strip()
                    ]

            pos += tag_len
            if tag == 255:
                break

        if recording_date is not None and recording_time is not None:
            try:
                recording_date = recording_date.replace(
                    hour=recording_time[0],
                    minute=recording_time[1],
                    second=recording_time[2],
                )
            except (ValueError, IndexError):
                pass

        return {
            "patient": info,
            "recording_date": recording_date,
            "device": device,
            "filters": filters,
            "interpretation": interpretation,
            "raw_metadata": extra,
        }

    def _parse_section3(self, data: bytes, offset: int, length: int) -> tuple[int, list[dict]]:
        """Parse Section 3: Lead definition."""
        header_size = 16
        pos = offset + header_size
        end = offset + length

        if pos + 2 > len(data):
            return 0, []

        num_leads = data[pos]
        flags = data[pos + 1] if pos + 1 < len(data) else 0
        pos += 2

        lead_defs = []
        for i in range(num_leads):
            if pos + 9 > len(data) or pos + 9 > end:
                break
            start_sample = _read_u32(data, pos)
            end_sample = _read_u32(data, pos + 4)
            lead_id = data[pos + 8]
            lead_defs.append({
                "start_sample": start_sample,
                "end_sample": end_sample,
                "lead_id": lead_id,
                "label": _SCP_LEAD_LABELS.get(lead_id, f"Lead{lead_id}"),
            })
            pos += 9

        return num_leads, lead_defs

    def _parse_section4(
        self, data: bytes, offset: int, length: int
    ) -> list[tuple[int, int]]:
        """Parse Section 4: QRS location data.

        Returns a list of ``(onset, offset)`` sample-index tuples for each
        QRS complex detected by the device.
        """
        header_size = 16
        pos = offset + header_size
        end = min(offset + length, len(data))

        if pos + 2 > end:
            return []

        num_qrs = _read_u16(data, pos)
        pos += 2

        locations: list[tuple[int, int]] = []
        for _ in range(num_qrs):
            if pos + 8 > end:
                break
            onset = _read_u32(data, pos)
            off = _read_u32(data, pos + 4)
            locations.append((onset, off))
            pos += 8

        return locations

    def _parse_section5(
        self,
        data: bytes,
        offset: int,
        length: int,
        lead_defs: list[dict],
        use_default_huffman: bool,
    ) -> list[Lead]:
        """Parse Section 5: Reference / median beat data.

        The encoding mirrors Section 6 (Huffman + difference encoding) but
        contains shorter template beats rather than full rhythm strips.
        """
        header_size = 16
        pos = offset + header_size
        end = min(offset + length, len(data))

        if pos + 6 > end:
            return []

        avm = _read_u16(data, pos)
        sample_time = _read_u16(data, pos + 2)
        encoding_flag = data[pos + 4]
        compression = data[pos + 5] if pos + 5 < end else 0

        pos += 6

        sampling_rate = int(1_000_000 / sample_time) if sample_time > 0 else 500

        num_leads = len(lead_defs) if lead_defs else 1
        lead_byte_lengths: list[int] = []
        for _ in range(num_leads):
            if pos + 2 > end:
                break
            lead_byte_lengths.append(_read_u16(data, pos))
            pos += 2

        leads: list[Lead] = []
        for i in range(len(lead_byte_lengths)):
            byte_len = lead_byte_lengths[i]
            if pos + byte_len > len(data):
                byte_len = len(data) - pos
            if byte_len <= 0:
                continue

            lead_data = data[pos: pos + byte_len]
            num_samples = byte_len // 2 if not use_default_huffman else byte_len * 4

            if use_default_huffman and compression == 0:
                diffs = _decode_huffman(lead_data, num_samples, use_default=True)
            else:
                diffs = []
                for j in range(0, min(len(lead_data), num_samples * 2), 2):
                    diffs.append(struct.unpack_from("<h", lead_data, j)[0])

            if encoding_flag == 1:
                samples_list = _reconstruct_second_difference(diffs)
            else:
                samples_list = _reconstruct_first_difference(diffs)

            avm_f = float(avm) if avm > 0 else 1.0
            res = avm_f / 1_000.0
            samples = np.array(samples_list, dtype=np.float64)

            label = lead_defs[i]["label"] if i < len(lead_defs) else f"Lead{i}"

            raw = res != 1.0
            leads.append(Lead(
                label=label,
                samples=samples,
                sampling_rate=sampling_rate,
                resolution=res,
                resolution_unit="uV",
                adc_resolution=avm_f,
                adc_resolution_unit="nV",
                units="" if raw else "uV",
                is_raw=raw,
            ))

            pos += lead_byte_lengths[i]

        return leads

    def _parse_section7(
        self, data: bytes, offset: int, length: int
    ) -> GlobalMeasurements:
        """Parse Section 7: Global ECG measurements."""
        header_size = 16
        pos = offset + header_size
        end = min(offset + length, len(data))

        gm = GlobalMeasurements()

        if pos + 6 > end:
            return gm

        gm.qrs_count = _read_u16(data, pos)
        gm.rr_interval = _read_u16(data, pos + 2)
        gm.heart_rate = _read_u16(data, pos + 4)
        pos += 6

        if pos + 2 <= end:
            val = _read_u16(data, pos)
            if val != 0xFFFF:
                gm.pr_interval = val
            pos += 2
        if pos + 2 <= end:
            val = _read_u16(data, pos)
            if val != 0xFFFF:
                gm.qrs_duration = val
            pos += 2
        if pos + 2 <= end:
            val = _read_u16(data, pos)
            if val != 0xFFFF:
                gm.qt_interval = val
            pos += 2
        if pos + 2 <= end:
            val = _read_u16(data, pos)
            if val != 0xFFFF:
                gm.qtc_bazett = val
            pos += 2
        if pos + 2 <= end:
            val = _read_i16(data, pos)
            if val != -32768:
                gm.p_axis = val
            pos += 2
        if pos + 2 <= end:
            val = _read_i16(data, pos)
            if val != -32768:
                gm.qrs_axis = val
            pos += 2
        if pos + 2 <= end:
            val = _read_i16(data, pos)
            if val != -32768:
                gm.t_axis = val
            pos += 2

        return gm

    def _parse_section8(
        self, data: bytes, offset: int, length: int
    ) -> Interpretation:
        """Parse Section 8: Textual diagnosis / interpretation."""
        header_size = 16
        pos = offset + header_size
        end = min(offset + length, len(data))

        interp = Interpretation()

        if pos + 1 > end:
            return interp

        confirmed_flag = data[pos]
        pos += 1

        if confirmed_flag == 1:
            interp.source = "confirmed"
        elif confirmed_flag == 0:
            interp.source = "machine"

        raw = data[pos:end]
        text = raw.decode("ascii", errors="replace")
        statements = [(s.strip(), "") for s in text.split("\x00") if s.strip()]
        interp.statements = statements

        return interp

    def _parse_section6(
        self,
        data: bytes,
        offset: int,
        length: int,
        lead_defs: list[dict],
        use_default_huffman: bool,
    ) -> list[Lead]:
        """Parse Section 6: Rhythm data."""
        header_size = 16
        pos = offset + header_size
        end = min(offset + length, len(data))

        if pos + 6 > end:
            return []

        avm = _read_u16(data, pos)
        sample_time = _read_u16(data, pos + 2)
        encoding_flag = data[pos + 4]
        compression = data[pos + 5] if pos + 5 < end else 0

        pos += 6

        sampling_rate = int(1_000_000 / sample_time) if sample_time > 0 else 500

        num_leads = len(lead_defs) if lead_defs else 1
        lead_byte_lengths: list[int] = []
        for i in range(num_leads):
            if pos + 2 > end:
                break
            lead_byte_lengths.append(_read_u16(data, pos))
            pos += 2

        leads: list[Lead] = []
        for i in range(len(lead_byte_lengths)):
            byte_len = lead_byte_lengths[i]
            if pos + byte_len > len(data):
                byte_len = len(data) - pos
            if byte_len <= 0:
                continue

            lead_data = data[pos: pos + byte_len]
            num_samples = lead_defs[i]["end_sample"] - lead_defs[i]["start_sample"] + 1 if i < len(lead_defs) else byte_len // 2

            if use_default_huffman and compression == 0:
                diffs = _decode_huffman(lead_data, num_samples, use_default=True)
            else:
                diffs = []
                for j in range(0, min(len(lead_data), num_samples * 2), 2):
                    diffs.append(struct.unpack_from("<h", lead_data, j)[0])

            if encoding_flag == 1:
                samples_list = _reconstruct_second_difference(diffs)
            else:
                samples_list = _reconstruct_first_difference(diffs)

            avm_f = float(avm) if avm > 0 else 1.0
            res = avm_f / 1_000.0
            samples = np.array(samples_list, dtype=np.float64)

            label = lead_defs[i]["label"] if i < len(lead_defs) else f"Lead{i}"

            raw = res != 1.0
            leads.append(Lead(
                label=label,
                samples=samples,
                sampling_rate=sampling_rate,
                resolution=res,
                resolution_unit="uV",
                adc_resolution=avm_f,
                adc_resolution_unit="nV",
                units="" if raw else "uV",
                is_raw=raw,
            ))

            pos += lead_byte_lengths[i]

        return leads
