"""Compression codecs for ECG data formats."""

from ecgdatakit.parsing.codecs.lzw import LzwDecoder
from ecgdatakit.parsing.codecs.xli import xli_decode

__all__ = ["LzwDecoder", "xli_decode"]
