"""ECG signal resampling utilities."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead, require_finite, require_scipy


def resample(lead: LeadLike, target_rate: int, *, fs: int | None = None) -> Lead:
    """Resample a lead to a different sample rate.

    Uses polyphase rational resampling (``scipy.signal.resample_poly``)
    which preserves signal content up to the Nyquist frequency of the
    lower sample rate.  The signal is extended linearly at both ends
    (``padtype="line"``) so a DC level or slow drift does not ring at the
    edges.  Output length is ``ceil(n * target_rate / fs)``.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or 1-D signal array (must be finite).
    target_rate : int
        Desired output sample rate in Hz.  A whole-number float such as
        ``250.0`` is accepted; the result is stored as an int.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    try:
        valid = math.isfinite(target_rate) and target_rate > 0
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"target_rate must be positive, got {target_rate!r}")
    if target_rate != int(target_rate):
        raise ValueError(
            f"target_rate must be a whole number of Hz, got {target_rate}"
        )
    target_rate = int(target_rate)
    require_finite(lead, "resample")
    if target_rate == lead.sampling_rate:
        return new_lead(lead, samples=lead.samples.copy(), sampling_rate=target_rate)

    sig = require_scipy("signal")

    # Exact ratio, also for a non-integer source rate (e.g. 499.5 Hz)
    ratio = Fraction(target_rate) / Fraction(lead.sampling_rate).limit_denominator(10**6)
    up, down = ratio.numerator, ratio.denominator

    resampled = sig.resample_poly(lead.samples, up, down, padtype="line").astype(np.float64)
    return new_lead(lead, samples=resampled, sampling_rate=target_rate)
