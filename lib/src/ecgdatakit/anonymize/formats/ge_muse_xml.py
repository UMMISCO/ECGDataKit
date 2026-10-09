"""GE MUSE XML anonymization.

Fields replaced: ``PatientDemographics/PatientID``, ``PatientLastName`` and
``PatientFirstName``, ``TestDemographics/SecondaryID`` (second patient ID)
and ``PharmaData/PharmaUniqueECGID`` (ECG ID). Waveform data is never
searched.
"""

from __future__ import annotations

from ecgdatakit.anonymize.formats._xml import Rule, XMLHandler


class MuseHandler(XMLHandler):
    parser = "GEMuseXMLParser"
    rules = (
        Rule(("PatientDemographics", "PatientID"), "patient_ids"),
        Rule(("PatientDemographics", "PatientLastName"), "last_names"),
        Rule(("PatientDemographics", "PatientFirstName"), "first_names"),
        Rule(("TestDemographics", "SecondaryID"), "patient_ids"),
        Rule(("PharmaData", "PharmaUniqueECGID"), "ecg_ids"),
    )
    skip_elements = frozenset({"waveformdata", "leaddata"})
