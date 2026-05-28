"""ECG signal resampling utilities."""

from __future__ import annotations

from math import gcd

import numpy as np

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead, require_scipy


def resample(lead: LeadLike, target_rate: int, *, fs: int | None = None) -> Lead:
    """Resample a lead to a different sample rate.

    Uses polyphase rational resampling (``scipy.signal.resample_poly``)
    which preserves signal content up to the Nyquist frequency of the
    lower sample rate.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    target_rate : int
        Desired output sample rate in Hz.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    if target_rate <= 0:
        raise ValueError(f"target_rate must be positive, got {target_rate}")
    if target_rate == lead.sampling_rate:
        return new_lead(lead, samples=lead.samples.copy())

    sig = require_scipy("signal")

    up = target_rate
    down = lead.sampling_rate
    divisor = gcd(up, down)
    up //= divisor
    down //= divisor

    resampled = sig.resample_poly(lead.samples, up, down).astype(np.float64)
    return new_lead(lead, samples=resampled, sampling_rate=target_rate)
