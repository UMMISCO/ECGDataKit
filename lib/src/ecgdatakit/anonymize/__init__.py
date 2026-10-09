"""Anonymization of ECG files.

Copies the ECG files of a source folder into ``ANONYMIZED/`` with the
patient's name and ID and the ECG ID replaced by pseudonyms, and keeps the
link in a CSV catalog. With ``datasets=True`` each sub-folder of the source
is handled separately. Raw files are only read.

    from ecgdatakit.anonymize import Anonymizer

    Anonymizer("/path/to/source").run()

Command line: ``ecgdatakit anonymize --help``. Needs ``shortuuid``
(``pip install 'ecgdatakit[anonymize]'``).
"""

from ecgdatakit.anonymize._catalog import COLUMNS, Catalog
from ecgdatakit.anonymize._engine import Anonymizer, DatasetLocked, Report
from ecgdatakit.anonymize._identity import Identity

__all__ = ["Anonymizer", "Catalog", "COLUMNS", "Identity", "DatasetLocked", "Report"]
