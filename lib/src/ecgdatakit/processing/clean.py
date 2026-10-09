"""ECG signal cleaning.

Methods
-------
default
    Built-in Butterworth bandpass (0.5--40 Hz) + power-line notch. No extra deps.
biosppy
    BioSPPy ECG filter (FIR band-pass, no notch). Requires ``pip install biosppy``.
neurokit2
    NeuroKit2 ``ecg_clean`` (0.5 Hz high-pass + power-line filter).
    Requires ``pip install neurokit2``.
combined
    BioSPPy followed by NeuroKit2. Requires both.
deepfade
    DeepFADE DenseNet encoder-decoder denoiser. Requires ``pip install torch``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead, require_finite

_DEEPFADE_WEIGHTS = Path(__file__).parent / "nn" / "weights" / "deepfade_exp_1_ddp.pt"


_METHOD_KWARGS: dict[str, frozenset[str]] = {
    "default": frozenset(),
    "biosppy": frozenset(),
    "neurokit2": frozenset(),
    "combined": frozenset(),
    "deepfade": frozenset({"weights_path", "device", "batch_size"}),
}


def clean_ecg(
    lead: LeadLike,
    method: str = "default",
    *,
    fs: int | None = None,
    powerline: float | None = 50.0,
    **kwargs,
) -> Lead:
    """Clean an ECG lead signal.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or 1-D signal array.
    method : str
        Cleaning method: ``"default"``, ``"biosppy"``, ``"neurokit2"``,
        ``"combined"``, or ``"deepfade"`` (experimental, warns on use).
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    powerline : float | None
        Power-line frequency in Hz (50 or 60) removed by ``"default"``,
        ``"neurokit2"`` and the NeuroKit2 stage of ``"combined"``.  ``None``
        disables the notch of ``"default"``.  BioSPPy and DeepFADE have no
        power-line stage and ignore it.  With ``"default"``, a notch that is
        not below the Nyquist frequency is skipped with a warning.
    **kwargs
        Extra arguments for ``"deepfade"`` only:

        - ``device`` (str): PyTorch device (default ``"cpu"``).
        - ``weights_path`` (str | Path): Override the bundled DeepFADE weights.
        - ``batch_size`` (int): Inference batch size (default 32).

    Returns
    -------
    Lead
        Cleaned lead (new object, original unchanged).

    Raises
    ------
    ValueError
        Unknown *method*, or the backend rejects the signal (e.g. too short
        or containing NaN).  Backend errors are never hidden.
    TypeError
        A keyword argument the selected method does not accept.
    """
    lead = ensure_lead(lead, fs=fs)
    if method not in _METHOD_KWARGS:
        raise ValueError(f"Unknown method {method!r}; choose from {tuple(_METHOD_KWARGS)}")
    unknown = set(kwargs) - _METHOD_KWARGS[method]
    if unknown:
        raise TypeError(
            f"clean_ecg(method={method!r}) got unexpected keyword argument(s): "
            f"{', '.join(sorted(unknown))}"
        )

    if method == "default":
        return _clean_default(lead, powerline)
    if method == "biosppy":
        return _clean_biosppy(lead, method)
    if method == "neurokit2":
        return _clean_neurokit2(lead, powerline, method)
    if method == "combined":
        return _clean_neurokit2(_clean_biosppy(lead, method), powerline, method)
    return _clean_deepfade(lead, **kwargs)


def _clean_default(lead: Lead, powerline: float | None) -> Lead:
    """Bandpass 0.5--40 Hz + power-line notch using built-in Butterworth filters."""
    from ecgdatakit.processing.filters import _band_top, _notch_if_possible, bandpass

    high = _band_top(40.0, lead.sampling_rate, "clean_ecg")
    result = bandpass(lead, low=0.5, high=high, order=4)
    return _notch_if_possible(result, powerline, "clean_ecg")


def _import_backend(module: str, method: str):
    """Import a cleaning backend, keeping the real import error in the message.

    A backend can be installed yet fail to import because one of its own
    dependencies is missing (e.g. biosppy 2.2.4 needs peakutils).
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"clean_ecg(method={method!r}) needs {module.split('.')[0]}, which "
            f"could not be imported ({type(exc).__name__}: {exc}). "
            'Install it with: pip install "ecgdatakit[cleaning]"'
        ) from exc


def _clean_biosppy(lead: Lead, method: str = "biosppy") -> Lead:
    """Clean using BioSPPy ECG filtering pipeline."""
    bio_ecg = _import_backend("biosppy.signals.ecg", method)
    require_finite(lead, "clean_ecg")
    result = bio_ecg.ecg(
        lead.samples,
        sampling_rate=lead.sampling_rate,
        show=False,
    )
    return new_lead(lead, samples=np.asarray(result["filtered"], dtype=np.float64))


def _clean_neurokit2(lead: Lead, powerline: float | None, method: str = "neurokit2") -> Lead:
    """Clean using NeuroKit2 ecg_clean pipeline."""
    nk = _import_backend("neurokit2", method)
    require_finite(lead, "clean_ecg")
    extra = {} if powerline is None else {"powerline": powerline}
    filtered = nk.ecg_clean(lead.samples, sampling_rate=lead.sampling_rate, **extra)
    return new_lead(lead, samples=np.asarray(filtered, dtype=np.float64))


def _clean_deepfade(
    lead: Lead,
    *,
    weights_path: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> Lead:
    """Denoise using the DeepFADE neural network."""
    from ecgdatakit.processing.denoise import denoise_deepfade

    if weights_path is None:
        weights_path = _DEEPFADE_WEIGHTS
    return denoise_deepfade(lead, weights_path=weights_path, device=device, batch_size=batch_size)
