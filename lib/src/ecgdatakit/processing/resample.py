"""ECG signal resampling utilities."""

from __future__ import annotations

from math import gcd

import numpy as np

from ecgdatakit.models import Lead
from ecgdatakit.processing._core import new_lead, require_scipy


def resample(lead: Lead, target_rate: int) -> Lead:
    """Resample a lead to a different sample rate.

    Uses polyphase rational resampling (``scipy.signal.resample_poly``)
    which preserves signal content up to the Nyquist frequency of the
    lower sample rate.

    Parameters
    ----------
    lead : Lead
        Input ECG lead.
    target_rate : int
        Desired output sample rate in Hz.
    """
    if target_rate <= 0:
        raise ValueError(f"target_rate must be positive, got {target_rate}")
    if target_rate == lead.sample_rate:
        return new_lead(lead, samples=lead.samples.copy())

    sig = require_scipy("signal")

    up = target_rate
    down = lead.sample_rate
    divisor = gcd(up, down)
    up //= divisor
    down //= divisor

    resampled = sig.resample_poly(lead.samples, up, down).astype(np.float64)
    return new_lead(lead, samples=resampled, sample_rate=target_rate)
