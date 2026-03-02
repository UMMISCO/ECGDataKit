"""ECG lead derivation utilities.

Derives missing leads from Einthoven's triangle and Goldberger's equations.
Pure numpy — no scipy required.
"""

from __future__ import annotations

import numpy as np

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead


def _check_compatible(a: Lead, b: Lead) -> None:
    if a.sample_rate != b.sample_rate:
        raise ValueError(
            f"Sample rates must match: {a.label}={a.sample_rate} Hz, "
            f"{b.label}={b.sample_rate} Hz"
        )
    if len(a.samples) != len(b.samples):
        raise ValueError(
            f"Sample counts must match: {a.label}={len(a.samples)}, "
            f"{b.label}={len(b.samples)}"
        )


def derive_lead_iii(
    lead_i: LeadLike,
    lead_ii: LeadLike,
    *,
    fs: int | None = None,
) -> Lead:
    """Derive Lead III from Leads I and II (Einthoven's law: III = II - I).

    Parameters
    ----------
    lead_i : Lead | NDArray[np.float64]
        Lead I signal.
    lead_ii : Lead | NDArray[np.float64]
        Lead II signal.
    fs : int | None
        Sample rate in Hz.  Required when passing numpy arrays.
    """
    lead_i = ensure_lead(lead_i, fs=fs, label="I")
    lead_ii = ensure_lead(lead_ii, fs=fs, label="II")
    _check_compatible(lead_i, lead_ii)
    samples = (lead_ii.samples - lead_i.samples).astype(np.float64)
    return new_lead(lead_i, samples=samples, label="III")


def derive_augmented(
    lead_i: LeadLike,
    lead_ii: LeadLike,
    *,
    fs: int | None = None,
) -> list[Lead]:
    """Derive augmented limb leads aVR, aVL, aVF from Leads I and II.

    Parameters
    ----------
    lead_i : Lead | NDArray[np.float64]
        Lead I signal.
    lead_ii : Lead | NDArray[np.float64]
        Lead II signal.
    fs : int | None
        Sample rate in Hz.  Required when passing numpy arrays.

    Returns
    -------
    list[Lead]
        [aVR, aVL, aVF] in that order.
    """
    lead_i = ensure_lead(lead_i, fs=fs, label="I")
    lead_ii = ensure_lead(lead_ii, fs=fs, label="II")
    _check_compatible(lead_i, lead_ii)
    i, ii = lead_i.samples, lead_ii.samples

    avr = (-(i + ii) / 2.0).astype(np.float64)
    avl = (i - ii / 2.0).astype(np.float64)
    avf = (ii - i / 2.0).astype(np.float64)

    return [
        new_lead(lead_i, samples=avr, label="aVR"),
        new_lead(lead_i, samples=avl, label="aVL"),
        new_lead(lead_i, samples=avf, label="aVF"),
    ]


def derive_standard_12(
    lead_i: LeadLike,
    lead_ii: LeadLike,
    v1: LeadLike,
    v2: LeadLike,
    v3: LeadLike,
    v4: LeadLike,
    v5: LeadLike,
    v6: LeadLike,
    *,
    fs: int | None = None,
) -> list[Lead]:
    """Assemble a full 12-lead ECG, deriving III, aVR, aVL, aVF.

    Parameters
    ----------
    lead_i, lead_ii : Lead | NDArray[np.float64]
        Limb leads I and II.
    v1..v6 : Lead | NDArray[np.float64]
        Precordial leads.
    fs : int | None
        Sample rate in Hz.  Required when passing numpy arrays.

    Returns
    -------
    list[Lead]
        12 leads in standard order: I, II, III, aVR, aVL, aVF, V1–V6.
    """
    lead_i = ensure_lead(lead_i, fs=fs, label="I")
    lead_ii = ensure_lead(lead_ii, fs=fs, label="II")
    v1 = ensure_lead(v1, fs=fs, label="V1")
    v2 = ensure_lead(v2, fs=fs, label="V2")
    v3 = ensure_lead(v3, fs=fs, label="V3")
    v4 = ensure_lead(v4, fs=fs, label="V4")
    v5 = ensure_lead(v5, fs=fs, label="V5")
    v6 = ensure_lead(v6, fs=fs, label="V6")
    iii = derive_lead_iii(lead_i, lead_ii)
    avr, avl, avf = derive_augmented(lead_i, lead_ii)
    return [lead_i, lead_ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6]


def find_lead(leads: list[Lead], label: str) -> Lead | None:
    """Find a lead by label (case-insensitive).

    Parameters
    ----------
    leads : list[Lead]
        List of leads to search.
    label : str
        Lead label to find (e.g., "II", "avl", "V1").
    """
    target = label.lower()
    for lead in leads:
        if lead.label.lower() == target:
            return lead
    return None
