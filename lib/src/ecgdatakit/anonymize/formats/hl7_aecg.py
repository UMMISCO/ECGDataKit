"""HL7 aECG anonymization.

Fields replaced (HL7 v3 Annotated ECG):

- ``subjectDemographicPerson/name``: ``family`` and plain-text name become
  the patient code, ``given`` is emptied. ``prefix`` and ``suffix`` are kept.
- ``trialSubject/id@extension``: patient code.
- ``AnnotatedECG/id``: ``root`` becomes a UUID derived from the ECG code,
  ``extension`` the ECG code.

Waveform ``digits`` are never searched.
"""

from __future__ import annotations

from ecgdatakit.anonymize.formats._xml import Rule, XMLHandler


class HL7Handler(XMLHandler):
    parser = "HL7aECGParser"
    rules = (
        Rule(("subjectDemographicPerson", "name", "family"), "last_names"),
        Rule(("subjectDemographicPerson", "name", "given"), "first_names"),
        Rule(("subjectDemographicPerson", "name", "prefix"), "skip"),
        Rule(("subjectDemographicPerson", "name", "suffix"), "skip"),
        # A plain-text name (no parts) is read as the last name by the parser
        Rule(("subjectDemographicPerson", "name"), "last_names"),
        Rule(("trialSubject", "id"), "patient_ids", attr="extension"),
        Rule(("AnnotatedECG", "id"), "ecg_ids", attr="root", anchored=True, uuid=True),
        Rule(("AnnotatedECG", "id"), "ecg_ids", attr="extension", anchored=True),
    )
    skip_elements = frozenset({"digits"})
