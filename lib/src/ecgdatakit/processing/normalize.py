"""ECG signal normalization utilities.

Pure numpy — no scipy required.
"""

from __future__ import annotations

import numpy as np

from ecgdatakit.models import Lead
from ecgdatakit.processing._core import new_lead


def normalize_minmax(lead: Lead) -> Lead:
    """Scale signal to the [-1, 1] range.

    Parameters
    ----------
    lead : Lead
        Input ECG lead.
    """
    samples = lead.samples
    vmin, vmax = samples.min(), samples.max()
    span = vmax - vmin
    if span == 0:
        return new_lead(lead, samples=np.zeros_like(samples))
    normalized = 2.0 * (samples - vmin) / span - 1.0
    return new_lead(lead, samples=normalized.astype(np.float64))


def normalize_zscore(lead: Lead) -> Lead:
    """Normalize signal to zero mean and unit variance (z-score).

    Parameters
    ----------
    lead : Lead
        Input ECG lead.
    """
    samples = lead.samples
    std = samples.std()
    if std == 0:
        return new_lead(lead, samples=np.zeros_like(samples))
    normalized = (samples - samples.mean()) / std
    return new_lead(lead, samples=normalized.astype(np.float64))


def normalize_amplitude(lead: Lead, target_mv: float = 1.0) -> Lead:
    """Scale signal so that its maximum absolute amplitude equals *target_mv*.

    Parameters
    ----------
    lead : Lead
        Input ECG lead.
    target_mv : float
        Target peak amplitude (default 1.0).
    """
    samples = lead.samples
    peak = np.abs(samples).max()
    if peak == 0:
        return new_lead(lead, samples=np.zeros_like(samples))
    normalized = samples * (target_mv / peak)
    return new_lead(lead, samples=normalized.astype(np.float64))
