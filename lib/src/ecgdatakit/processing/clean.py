"""ECG signal cleaning.

Methods
-------
default
    Built-in Butterworth bandpass (0.5--40 Hz) + 50 Hz notch. No extra deps.
biosppy
    BioSPPy ECG filter. Requires ``pip install biosppy``.
neurokit2
    NeuroKit2 adaptive pipeline. Requires ``pip install neurokit2``.
combined
    BioSPPy followed by NeuroKit2. Requires both.
deepfade
    DeepFADE DenseNet encoder-decoder denoiser. Requires ``pip install torch``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead, require_scipy

_DEEPFADE_WEIGHTS = Path(__file__).parent / "nn" / "weights" / "deepfade_exp_1_ddp.pt"


def clean_ecg(lead: LeadLike, method: str = "default", *, fs: int | None = None, **kwargs) -> Lead:
    """Clean an ECG lead signal.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    method : str
        Cleaning method: ``"default"``, ``"biosppy"``, ``"neurokit2"``,
        ``"combined"``, or ``"deepfade"``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    **kwargs
        Extra arguments forwarded to the selected backend:

        - ``device`` (str): PyTorch device for ``"deepfade"`` (default ``"cpu"``).
        - ``weights_path`` (str | Path): Override the bundled DeepFADE weights.
        - ``batch_size`` (int): Inference batch size for ``"deepfade"`` (default 32).

    Returns
    -------
    Lead
        Cleaned lead (new object, original unchanged).
    """
    lead = ensure_lead(lead, fs=fs)
    methods = ("default", "biosppy", "neurokit2", "combined", "deepfade")
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}; choose from {methods}")

    dispatch = {
        "default": _clean_default,
        "biosppy": _clean_biosppy,
        "neurokit2": _clean_neurokit2,
        "combined": _clean_combined,
        "deepfade": _clean_deepfade,
    }
    return dispatch[method](lead, **kwargs)


def _clean_default(lead: Lead, **_kwargs) -> Lead:
    """Bandpass 0.5--40 Hz + 50 Hz notch using built-in Butterworth filters."""
    from ecgdatakit.processing.filters import bandpass, notch

    result = bandpass(lead, low=0.5, high=40.0, order=4)
    return notch(result, freq=50.0, quality=30.0)


def _require_biosppy():
    try:
        import biosppy
        return biosppy
    except ImportError as exc:
        raise ImportError(
            "biosppy is required for method='biosppy'. "
            "Install it with: pip install biosppy"
        ) from exc


def _clean_biosppy(lead: Lead, **_kwargs) -> Lead:
    """Clean using BioSPPy ECG filtering pipeline."""
    bio = _require_biosppy()
    try:
        result = bio.signals.ecg.ecg(
            lead.samples,
            sampling_rate=lead.sample_rate,
            show=False,
        )
        filtered = result["filtered"].astype(np.float64)
    except (ValueError, Exception):
        filtered = lead.samples.copy()
    return new_lead(lead, samples=filtered)


def _require_neurokit2():
    try:
        import neurokit2 as nk
        return nk
    except ImportError as exc:
        raise ImportError(
            "neurokit2 is required for method='neurokit2'. "
            "Install it with: pip install neurokit2"
        ) from exc


def _clean_neurokit2(lead: Lead, **_kwargs) -> Lead:
    """Clean using NeuroKit2 ecg_clean pipeline."""
    nk = _require_neurokit2()
    try:
        filtered = nk.ecg_clean(
            lead.samples,
            sampling_rate=lead.sample_rate,
        ).astype(np.float64)
    except (ValueError, Exception):
        filtered = lead.samples.copy()
    return new_lead(lead, samples=filtered)


def _clean_combined(lead: Lead, **kwargs) -> Lead:
    """Clean using BioSPPy then NeuroKit2 sequentially."""
    return _clean_neurokit2(_clean_biosppy(lead, **kwargs), **kwargs)


def _clean_deepfade(
    lead: Lead,
    *,
    weights_path: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    **_kwargs,
) -> Lead:
    """Denoise using the DeepFADE neural network."""
    from ecgdatakit.processing.denoise import denoise_deepfade

    if weights_path is None:
        weights_path = _DEEPFADE_WEIGHTS
    return denoise_deepfade(lead, weights_path=weights_path, device=device, batch_size=batch_size)
