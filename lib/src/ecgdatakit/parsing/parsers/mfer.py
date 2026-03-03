"""MFER (Medical waveform Format Encoding Rules) parser.

Reference: ISO 22077 — TLV binary format for medical waveforms.
"""

from __future__ import annotations

import struct
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
)
from ecgdatakit.parsing.parser import Parser

_TAG_ENDIAN = 0x01
_TAG_WAVEFORM_TYPE = 0x03
_TAG_SAMPLING_RATE = 0x04
_TAG_NUM_CHANNELS = 0x05
_TAG_NUM_SAMPLES = 0x06
_TAG_DATA_BLOCK = 0x09
_TAG_RESOLUTION = 0x0B
_TAG_CHANNEL_DEF = 0x0A
_TAG_DATA_TYPE = 0x02
_TAG_COMMENT = 0x07
_TAG_FILTER = 0x08
_TAG_BLOCK_INFO = 0x0C
_TAG_CHANNEL_EXT = 0x0D
_TAG_PATIENT_ID = 0x31
_TAG_PATIENT_NAME = 0x32
_TAG_BIRTH_DATE = 0x33
_TAG_SEX = 0x34
_TAG_COMMENT_2 = 0x3F
_TAG_INSTITUTION = 0x40
_TAG_RECORD_TIME = 0x41
_TAG_DEPARTMENT = 0x41
_TAG_TECHNICIAN = 0x44
_TAG_EVENT = 0x45


def _read_ber_length(data: bytes, offset: int) -> tuple[int, int]:
    """Read BER-style length encoding.

    Returns (length_value, bytes_consumed).
    Short form: 1 byte if < 128.
    Long form: first byte = 0x80 | num_length_bytes, then num_length_bytes of length.
    """
    if offset >= len(data):
        return 0, 0

    first = data[offset]
    if first < 0x80:
        return first, 1
    elif first == 0x80:
        return 0, 1
    else:
        num_bytes = first & 0x7F
        if offset + 1 + num_bytes > len(data):
            return 0, 1
        length = 0
        for i in range(num_bytes):
            length = (length << 8) | data[offset + 1 + i]
        return length, 1 + num_bytes


def _read_mfer_int(data: bytes, length: int) -> int:
    """Read a big-endian signed integer of variable length."""
    if length == 0:
        return 0
    val = int.from_bytes(data[:length], byteorder="big", signed=True)
    return val


def _read_mfer_uint(data: bytes, length: int) -> int:
    """Read a big-endian unsigned integer of variable length."""
    if length == 0:
        return 0
    return int.from_bytes(data[:length], byteorder="big", signed=False)


class MFERParser(Parser):
    """Parser for MFER binary ECG files (ISO 22077)."""

    FORMAT_NAME = "MFER"
    FORMAT_DESCRIPTION = "Medical waveform Format Encoding Rules (ISO 22077)"
    FILE_EXTENSIONS = [".mfer", ".mfr"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if len(header) < 6:
            return False

        first_byte = header[0]
        if first_byte < 0x01 or first_byte > 0x1F:
            return False

        try:
            length, consumed = _read_ber_length(header, 1)
            if consumed == 0 or length < 0 or length > 100_000_000:
                return False
            next_offset = 1 + consumed + length
            if next_offset >= len(header):
                return False
            next_tag = header[next_offset]
            if next_tag < 0x01 or next_tag > 0x7F:
                return False
            length2, consumed2 = _read_ber_length(header, next_offset + 1)
            if consumed2 == 0 or length2 < 0 or length2 > 100_000_000:
                return False
            return True
        except Exception:
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            data = f.read()

        if len(data) < 4:
            raise CorruptedFileError(f"File too small for MFER: {len(data)} bytes")

        record = ECGRecord(source_format="mfer")

        big_endian = True
        num_channels = 1
        num_samples = 0
        sample_interval_us = 1000
        resolution = 1.0
        data_type = 0
        data_block: bytes | None = None
        patient = PatientInfo()
        recording = RecordingInfo()
        device_info = DeviceInfo()
        filters = FilterSettings()
        channel_labels: list[str] = []
        channel_units: list[str] = []
        raw_metadata: dict = {}
        comments: list[str] = []
        events: list[dict] = []

        offset = 0
        while offset < len(data):
            if offset >= len(data):
                break

            tag = data[offset]
            offset += 1

            if tag == 0x00 or tag > 0x7F:
                break

            length, consumed = _read_ber_length(data, offset)
            offset += consumed

            if offset + length > len(data):
                break

            tag_data = data[offset: offset + length]

            if tag == _TAG_ENDIAN and length >= 1:
                big_endian = (tag_data[0] == 0)

            elif tag == _TAG_DATA_TYPE and length >= 1:
                data_type = tag_data[0]

            elif tag == _TAG_WAVEFORM_TYPE and length >= 1:
                raw_metadata["waveform_type"] = _read_mfer_uint(tag_data, length)

            elif tag == _TAG_COMMENT and length >= 1:
                raw_metadata["comment"] = tag_data.decode("utf-8", errors="replace").strip("\x00 ")

            elif tag == _TAG_FILTER and length >= 1:
                if length >= 12:
                    filters.highpass = struct.unpack(">f", tag_data[:4])[0]
                    filters.lowpass = struct.unpack(">f", tag_data[4:8])[0]
                    notch_val = struct.unpack(">f", tag_data[8:12])[0]
                    if notch_val > 0:
                        filters.notch = notch_val
                        filters.notch_active = True
                elif length >= 8:
                    filters.highpass = struct.unpack(">f", tag_data[:4])[0]
                    filters.lowpass = struct.unpack(">f", tag_data[4:8])[0]
                elif length >= 4:
                    filters.lowpass = struct.unpack(">f", tag_data[:4])[0]
                elif length >= 2:
                    filters.lowpass = float(struct.unpack(">H", tag_data[:2])[0])

            elif tag == _TAG_SAMPLING_RATE and length >= 1:
                if length >= 4:
                    sample_interval_us = struct.unpack(">i", tag_data[:4])[0]
                elif length >= 2:
                    sample_interval_us = struct.unpack(">h", tag_data[:2])[0]
                else:
                    sample_interval_us = tag_data[0]
                if sample_interval_us <= 0:
                    sample_interval_us = 1000

            elif tag == _TAG_NUM_CHANNELS and length >= 1:
                num_channels = _read_mfer_uint(tag_data, length)
                if num_channels <= 0:
                    num_channels = 1

            elif tag == _TAG_NUM_SAMPLES and length >= 1:
                num_samples = _read_mfer_uint(tag_data, length)

            elif tag == _TAG_RESOLUTION and length >= 1:
                if length >= 4:
                    resolution = struct.unpack(">f", tag_data[:4])[0]
                elif length >= 2:
                    resolution = struct.unpack(">h", tag_data[:2])[0]
                else:
                    resolution = float(tag_data[0])

            elif tag == _TAG_DATA_BLOCK:
                data_block = tag_data

            elif tag == _TAG_BLOCK_INFO and length >= 1:
                raw_metadata["block_info"] = tag_data.hex()

            elif tag == _TAG_CHANNEL_EXT and length >= 1:
                unit_str = tag_data.decode("ascii", errors="replace").strip("\x00 ")
                if unit_str:
                    channel_units.append(unit_str)

            elif tag == _TAG_CHANNEL_DEF and length >= 1:
                label = tag_data.decode("ascii", errors="replace").strip("\x00 ")
                if label:
                    channel_labels.append(label)

            elif tag == _TAG_PATIENT_ID and length >= 1:
                patient.patient_id = tag_data.decode("ascii", errors="replace").strip("\x00 ")

            elif tag == _TAG_PATIENT_NAME and length >= 1:
                name = tag_data.decode("utf-8", errors="replace").strip("\x00 ")
                if "^" in name:
                    parts = name.split("^")
                    patient.last_name = parts[0].strip()
                    if len(parts) > 1:
                        patient.first_name = parts[1].strip()
                else:
                    patient.last_name = name

            elif tag == _TAG_BIRTH_DATE and length >= 4:
                try:
                    year = struct.unpack(">H", tag_data[:2])[0]
                    month = tag_data[2]
                    day = tag_data[3]
                    if 1 <= month <= 12 and 1 <= day <= 31 and year > 0:
                        patient.birth_date = datetime(year, month, day)
                except (ValueError, struct.error):
                    pass

            elif tag == _TAG_SEX and length >= 1:
                sex_code = tag_data[0]
                patient.sex = "M" if sex_code == 1 else "F" if sex_code == 2 else "U"

            elif tag == _TAG_COMMENT_2 and length >= 1:
                comment_text = tag_data.decode("utf-8", errors="replace").strip("\x00 ")
                if comment_text:
                    comments.append(comment_text)

            elif tag == _TAG_INSTITUTION and length >= 1:
                device_info.institution = tag_data.decode("utf-8", errors="replace").strip("\x00 ")

            elif tag == _TAG_RECORD_TIME and length >= 1:
                parsed_date = False
                if length >= 4:
                    try:
                        year = struct.unpack(">H", tag_data[:2])[0]
                        month = tag_data[2]
                        day = tag_data[3]
                        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                            if length >= 7:
                                hour = tag_data[4]
                                minute = tag_data[5]
                                second = tag_data[6]
                                recording.date = datetime(year, month, day, hour, minute, second)
                            else:
                                recording.date = datetime(year, month, day)
                            parsed_date = True
                    except (ValueError, struct.error):
                        pass
                if not parsed_date:
                    dept = tag_data.decode("utf-8", errors="replace").strip("\x00 ")
                    if dept:
                        device_info.department = dept

            elif tag == _TAG_TECHNICIAN and length >= 1:
                recording.technician = tag_data.decode("utf-8", errors="replace").strip("\x00 ")

            elif tag == _TAG_EVENT and length >= 1:
                event_entry: dict = {"raw": tag_data.hex()}
                try:
                    event_entry["description"] = tag_data.decode("utf-8", errors="replace").strip("\x00 ")
                except Exception:
                    pass
                events.append(event_entry)

            offset += length

        sample_rate = int(1_000_000 / sample_interval_us) if sample_interval_us > 0 else 1000
        recording.sample_rate = sample_rate

        record.patient = patient
        record.recording = recording
        record.device = device_info
        record.filters = filters

        if comments:
            raw_metadata["comments"] = comments
        if events:
            raw_metadata["events"] = events

        leads: list[Lead] = []
        if data_block is not None and num_channels > 0:
            if data_type == 0:
                dtype_str = ">i2" if big_endian else "<i2"
            elif data_type == 1:
                dtype_str = ">u2" if big_endian else "<u2"
            elif data_type == 2:
                dtype_str = ">i4" if big_endian else "<i4"
            else:
                dtype_str = ">i2" if big_endian else "<i2"

            item_size = np.dtype(dtype_str).itemsize
            total_items = len(data_block) // item_size
            raw = np.frombuffer(data_block[:total_items * item_size], dtype=dtype_str)

            if num_samples == 0 and num_channels > 0:
                num_samples = total_items // num_channels

            usable = num_samples * num_channels
            raw = raw[:usable]

            if len(raw) >= usable and usable > 0:
                channels = raw.reshape((num_samples, num_channels))

                for ch in range(num_channels):
                    label = channel_labels[ch] if ch < len(channel_labels) else f"Ch{ch + 1}"
                    units = channel_units[ch] if ch < len(channel_units) else ""
                    samples = channels[:, ch].astype(np.float64) * resolution

                    leads.append(Lead(
                        label=label,
                        samples=samples,
                        sample_rate=sample_rate,
                        resolution=resolution,
                        units=units,
                    ))

        record.leads = leads

        if leads and sample_rate > 0:
            duration_s = len(leads[0].samples) / sample_rate
            recording.duration = timedelta(seconds=duration_s)

        # Signal characteristics from data_type
        if data_type == 0:
            sig_bits, sig_signed, sig_encoding = 16, True, "int16"
        elif data_type == 1:
            sig_bits, sig_signed, sig_encoding = 16, False, "uint16"
        elif data_type == 2:
            sig_bits, sig_signed, sig_encoding = 32, True, "int32"
        else:
            sig_bits, sig_signed, sig_encoding = 16, True, "int16"

        record.signal = SignalCharacteristics(
            bits_per_sample=sig_bits,
            signal_signed=sig_signed,
            number_channels_allocated=num_channels,
            number_channels_valid=len(leads),
            data_encoding=sig_encoding,
            compression="none",
        )

        record.raw_metadata.update(raw_metadata)
        record.raw_metadata["filepath"] = str(file_path)
        record.raw_metadata["big_endian"] = big_endian
        record.raw_metadata["data_type"] = data_type

        return record
