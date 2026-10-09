"""Philips Sierra XML anonymization.

Fields replaced, under ``patient/generalpatientdata``: ``patientid``,
``uniquepatientid``, ``MRN``, ``name/lastname``, ``name/firstname``,
``name/middlename`` (and the same names directly under
``generalpatientdata``). ``documentinfo/documentname`` is the ECG ID.
Encoded waveforms are never searched.
"""

from __future__ import annotations

from ecgdatakit.anonymize.formats._xml import Rule, XMLHandler


class SierraHandler(XMLHandler):
    parser = "SierraXMLParser"
    rules = (
        Rule(("generalpatientdata", "patientid"), "patient_ids"),
        Rule(("generalpatientdata", "uniquepatientid"), "patient_ids"),
        Rule(("generalpatientdata", "MRN"), "patient_ids"),
        Rule(("generalpatientdata", "name", "lastname"), "last_names"),
        Rule(("generalpatientdata", "name", "firstname"), "first_names"),
        Rule(("generalpatientdata", "name", "middlename"), "first_names"),
        Rule(("generalpatientdata", "lastname"), "last_names"),
        Rule(("generalpatientdata", "firstname"), "first_names"),
        Rule(("documentinfo", "documentname"), "ecg_ids"),
    )
    skip_elements = frozenset({"parsedwaveforms", "waveforms", "repbeats"})
