"""Record-level helpers shared by all parsers."""

from __future__ import annotations

from ecgdatakit.models import ECGRecord, _TO_UV, _normalize_unit


def fill_signal_summary(record: ECGRecord) -> None:
    """Fill record-level sampling rate and resolution from the leads.

    ``acquisition.signal.sampling_rate`` is set when it is 0 and all leads
    share one rate. ``acquisition.signal.resolution`` (uV per count) is set
    when it is 0 and all leads share one resolution in a voltage unit.
    """
    signal = record.recording.acquisition.signal
    leads = record.leads
    if not leads:
        return
    rates = {lead.sampling_rate for lead in leads}
    if not signal.sampling_rate and len(rates) == 1:
        signal.sampling_rate = rates.pop()
    if not signal.resolution:
        values = set()
        for lead in leads:
            unit = _normalize_unit(lead.resolution_unit) if lead.resolution_unit else None
            if unit is None:
                return
            values.add(round(lead.resolution * _TO_UV[unit], 9))
        if len(values) == 1:
            signal.resolution = values.pop()
