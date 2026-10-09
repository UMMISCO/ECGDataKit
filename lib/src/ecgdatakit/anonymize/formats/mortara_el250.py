"""Mortara ELI XML anonymization.

Fields replaced: ``SUBJECT`` attributes ``LAST_NAME``, ``FIRST_NAME``,
``MIDDLE_NAME`` and ``ID``, and the ``VALUE`` of ``DEMOGRAPHIC_FIELD``
codes 1 (last name), 7 (first name) and 2 (patient ID). Mortara files have
no ECG ID. Base64 ``DATA`` attributes are never searched.
"""

from __future__ import annotations

from ecgdatakit.anonymize.formats._xml import Rule, XMLHandler


class MortaraHandler(XMLHandler):
    parser = "MortaraEL250Parser"
    rules = (
        Rule(("SUBJECT",), "last_names", attr="LAST_NAME"),
        Rule(("SUBJECT",), "first_names", attr="FIRST_NAME"),
        Rule(("SUBJECT",), "first_names", attr="MIDDLE_NAME"),
        Rule(("SUBJECT",), "patient_ids", attr="ID"),
        # DEMOGRAPHIC_FIELD codes: 1 last name, 7 first name, 2 patient ID
        Rule(("DEMOGRAPHIC_FIELD",), "last_names", attr="VALUE", when=("ID", "1")),
        Rule(("DEMOGRAPHIC_FIELD",), "first_names", attr="VALUE", when=("ID", "7")),
        Rule(("DEMOGRAPHIC_FIELD",), "patient_ids", attr="VALUE", when=("ID", "2")),
    )
    skip_attrs = frozenset({"data"})
