"""ECG lead derivation utilities.

Derives missing leads from Einthoven's triangle and Goldberger's equations.
Pure numpy, no scipy required.

Scaling: raw leads that share the same ``resolution`` and
``resolution_unit`` are combined in ADC counts (offsets folded in) and the
derived leads stay raw on that scale.  Otherwise both leads are converted
to physical values (in lead I's unit) first, and the derived leads are
physical.  Raw leads with different scales and no voltage unit raise
``ValueError``.  Derived leads carry no file-specific metadata
(``annotations``, ``quality``, ``transducer``, ``adc_resolution`` are reset).
"""

from __future__ import annotations

import numpy as np

from ecgdatakit.models import Lead, LeadLike, _normalize_unit, derive_is_raw
from ecgdatakit.processing._core import ensure_lead, fold_offset, new_lead


def _check_compatible(a: Lead, b: Lead) -> None:
    if a.sampling_rate != b.sampling_rate:
        raise ValueError(
            f"Sample rates must match: {a.label}={a.sampling_rate} Hz, "
            f"{b.label}={b.sampling_rate} Hz"
        )
    if len(a.samples) != len(b.samples):
        raise ValueError(
            f"Sample counts must match: {a.label}={len(a.samples)}, "
            f"{b.label}={len(b.samples)}"
        )


def _to_physical(lead: Lead) -> Lead:
    if not lead.is_raw:
        return lead
    if not _normalize_unit(lead.resolution_unit) or lead.resolution == 0.0:
        raise ValueError(
            f"Lead {lead.label!r} is raw ADC counts without a usable voltage "
            "scale, and its scale differs from the other lead; cannot "
            "combine them"
        )
    return lead.to_physical()


def _common_scale(a: Lead, b: Lead) -> tuple[Lead, Lead]:
    """Bring two leads onto one scale so samples can be added.

    Raw leads with identical ``resolution`` and ``resolution_unit`` stay in
    ADC counts (offsets folded into the samples).  Any other combination is
    converted to physical values in lead *a*'s unit.
    """
    if (
        a.is_raw and b.is_raw
        and a.resolution == b.resolution
        and a.resolution_unit == b.resolution_unit
    ):
        if a.offset != b.offset and a.resolution == 0.0:
            raise ValueError(
                f"Leads {a.label!r} and {b.label!r} have different offsets "
                "and no resolution; cannot combine them"
            )
        return fold_offset(a), fold_offset(b)
    a, b = _to_physical(a), _to_physical(b)
    ua, ub = _normalize_unit(a.units), _normalize_unit(b.units)
    if ua and ub:
        if ua != ub:
            b = b.convert_units(a.units)
    elif a.units != b.units:
        raise ValueError(
            f"Leads have incompatible units: {a.label}={a.units!r}, "
            f"{b.label}={b.units!r}"
        )
    return a, b


def _derived(template: Lead, other: Lead, samples, label: str) -> Lead:
    """Lead computed from *template* and *other*, without file-specific metadata."""
    if template.is_raw:
        scale = dict(
            resolution=template.resolution,
            resolution_unit=template.resolution_unit,
            is_raw=derive_is_raw(template.resolution, 0.0, template.resolution_unit),
        )
    else:
        scale = dict(resolution=1.0, resolution_unit=template.units, is_raw=False)
    return new_lead(
        template,
        samples=np.asarray(samples, dtype=np.float64),
        label=label,
        offset=0.0,
        adc_resolution=0.0,
        adc_resolution_unit="",
        quality=None,
        transducer="",
        prefiltering=(
            template.prefiltering
            if template.prefiltering == other.prefiltering else ""
        ),
        annotations={},
        **scale,
    )


def _prepare_limb(lead_i: LeadLike, lead_ii: LeadLike, fs: int | None) -> tuple[Lead, Lead]:
    lead_i = ensure_lead(lead_i, fs=fs, label="I")
    lead_ii = ensure_lead(lead_ii, fs=fs, label="II")
    _check_compatible(lead_i, lead_ii)
    return _common_scale(lead_i, lead_ii)


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
    lead_i, lead_ii = _prepare_limb(lead_i, lead_ii, fs)
    return _derived(lead_i, lead_ii, lead_ii.samples - lead_i.samples, "III")


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
    lead_i, lead_ii = _prepare_limb(lead_i, lead_ii, fs)
    i, ii = lead_i.samples, lead_ii.samples
    return [
        _derived(lead_i, lead_ii, -(i + ii) / 2.0, "aVR"),
        _derived(lead_i, lead_ii, i - ii / 2.0, "aVL"),
        _derived(lead_i, lead_ii, ii - i / 2.0, "aVF"),
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
        When the limb leads had to be converted to physical values (see the
        module notes), I, II and V1–V6 are returned converted as well, so
        all 12 leads share one scale.
    """
    lead_i, lead_ii = _prepare_limb(lead_i, lead_ii, fs)
    precordial = []
    for lead, label in zip((v1, v2, v3, v4, v5, v6), ("V1", "V2", "V3", "V4", "V5", "V6")):
        lead = ensure_lead(lead, fs=fs, label=label)
        _check_compatible(lead_i, lead)
        if not lead_i.is_raw:
            # Limb leads were scaled to physical values; match them
            lead = _to_physical(lead)
            unit, target = _normalize_unit(lead.units), _normalize_unit(lead_i.units)
            if unit and target and unit != target:
                lead = lead.convert_units(lead_i.units)
        precordial.append(lead)
    i, ii = lead_i.samples, lead_ii.samples
    derived = [
        _derived(lead_i, lead_ii, ii - i, "III"),
        _derived(lead_i, lead_ii, -(i + ii) / 2.0, "aVR"),
        _derived(lead_i, lead_ii, i - ii / 2.0, "aVL"),
        _derived(lead_i, lead_ii, ii - i / 2.0, "aVF"),
    ]
    return [lead_i, lead_ii, *derived, *precordial]


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
