"""ECG lead derivation utilities.

Derives missing leads from Einthoven's triangle and Goldberger's equations.
Pure numpy — no scipy required.
"""

from __future__ import annotations

import numpy as np

from ecgdatakit.models import Lead
from ecgdatakit.processing._core import new_lead


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


def derive_lead_iii(lead_i: Lead, lead_ii: Lead) -> Lead:
    """Derive Lead III from Leads I and II (Einthoven's law: III = II - I).

    Parameters
    ----------
    lead_i : Lead
        Lead I signal.
    lead_ii : Lead
        Lead II signal.
    """
    _check_compatible(lead_i, lead_ii)
    samples = (lead_ii.samples - lead_i.samples).astype(np.float64)
    return new_lead(lead_i, samples=samples, label="III")


def derive_augmented(lead_i: Lead, lead_ii: Lead) -> list[Lead]:
    """Derive augmented limb leads aVR, aVL, aVF from Leads I and II.

    Returns
    -------
    list[Lead]
        [aVR, aVL, aVF] in that order.
    """
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
    lead_i: Lead,
    lead_ii: Lead,
    v1: Lead,
    v2: Lead,
    v3: Lead,
    v4: Lead,
    v5: Lead,
    v6: Lead,
) -> list[Lead]:
    """Assemble a full 12-lead ECG, deriving III, aVR, aVL, aVF.

    Parameters
    ----------
    lead_i, lead_ii : Lead
        Limb leads I and II.
    v1..v6 : Lead
        Precordial leads.

    Returns
    -------
    list[Lead]
        12 leads in standard order: I, II, III, aVR, aVL, aVF, V1–V6.
    """
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
