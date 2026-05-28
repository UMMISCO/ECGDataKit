"""ECG file parsing subpackage.

Contains all format-specific parsers, codecs, and XML helpers.
"""

from ecgdatakit.parsing.batch import parse_batch
from ecgdatakit.parsing.parser import FileParser, Parser

__all__ = ["FileParser", "Parser", "parse_batch"]
