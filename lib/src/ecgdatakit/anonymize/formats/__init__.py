"""Format handlers, one per ecgdatakit parser.

EDAN Holter is not handled: the format is undocumented.
"""

from __future__ import annotations

from ecgdatakit.anonymize.formats._base import Codes, Handler, Rewrite, Spliced
from ecgdatakit.anonymize.formats.alivecor_kardia import AliveCorHandler
from ecgdatakit.anonymize.formats.dicom_waveform import DICOMHandler
from ecgdatakit.anonymize.formats.edf import EDFHandler
from ecgdatakit.anonymize.formats.ge_muse_xml import MuseHandler
from ecgdatakit.anonymize.formats.hl7_aecg import HL7Handler
from ecgdatakit.anonymize.formats.ishne_holter import ISHNEHandler
from ecgdatakit.anonymize.formats.mfer import MFERHandler
from ecgdatakit.anonymize.formats.mortara_el250 import MortaraHandler
from ecgdatakit.anonymize.formats.scp_ecg import SCPHandler
from ecgdatakit.anonymize.formats.sierra_xml import SierraHandler
from ecgdatakit.anonymize.formats.wfdb import WFDBHandler

HANDLERS: dict[str, Handler] = {
    h.parser: h
    for h in (
        HL7Handler(), MuseHandler(), SierraHandler(), MortaraHandler(),
        ISHNEHandler(), EDFHandler(), AliveCorHandler(),
        SCPHandler(), MFERHandler(), WFDBHandler(), DICOMHandler(),
    )
}
"""Handler by parser class name."""

__all__ = ["HANDLERS", "Codes", "Handler", "Rewrite", "Spliced", "WFDBHandler"]
