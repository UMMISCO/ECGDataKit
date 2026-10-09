"""ECG denoising using the DeepFADE neural network (experimental).

DeepFADE is a denoising autoencoder developed as part of ECGDataKit,
trained on a large private multi-source ECG database with extensive
noise augmentations (baseline wander, electrode motion, muscle artifacts,
powerline interference).  The architecture is a symmetric DenseNet
encoder-decoder: the encoder compresses a 10-second single-lead segment
(500 Hz, 5 000 samples) through four dense blocks into an 8-channel
latent space, and the decoder mirrors the path to produce two outputs
— the denoised signal and the estimated baseline wander.

The network works on standardised segments and its output has no fixed
amplitude scale.  :func:`denoise_deepfade` therefore z-scores each
segment on the way in and rescales the output by a least-squares fit to
the 0.5--40 Hz band of the input, so amplitudes stay in the input units.
The model has not been validated on public benchmarks; treat its output
as experimental.

Pre-trained weights are bundled with the package.

Requires: ``pip install "ecgdatakit[denoising]"`` (torch >= 2.0)
"""

from __future__ import annotations

import re
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead, LeadLike
from ecgdatakit.processing._core import ensure_lead, new_lead, require_scipy

_EXPECTED_FS = 500
_EXPECTED_LEN = 5000
_HOP = _EXPECTED_LEN // 2
_MIN_DURATION_S = 1.0


def _require_torch():
    """Lazily import torch, raising a helpful error if missing."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for DeepFADE denoising. "
            'Install it with: pip install "ecgdatakit[denoising]"'
        ) from exc


def _external_stacklevel() -> int:
    """Stack level of the first caller outside the ecgdatakit package."""
    level = 1
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__", "").startswith("ecgdatakit"):
        frame = frame.f_back
        level += 1
    return level


def _remap_state_dict_key(key: str) -> str:
    """Remap legacy state-dict keys to match the current model architecture.

    Handles three transformations:
    - DDP ``module.`` prefix removal
    - Python name-mangling reversal (``_ClassName__attr`` → ``_attr``)
    - Named submodule to ModuleList index mapping
    """
    key = key.removeprefix("module.")
    key = re.sub(r"_[A-Z][A-Za-z0-9]*__", "_", key)
    key = key.replace("._dense_trunk.", "._trunk.")

    def _block_index(m: re.Match) -> str:
        idx = int(m.group(2))
        if m.group(1) == "DenseBlock":
            return f"_blocks.{2 * idx}"
        return f"_blocks.{2 * idx + 1}"

    return re.sub(r"(DenseBlock|TransitionBlock)_(\d+)", _block_index, key)


def _load_model(weights_path: str | Path, device: str = "cpu"):
    """Load a DeepFADE model with pre-trained weights (cached per path and device).

    Parameters
    ----------
    weights_path : str | Path
        Path to the ``.pt`` weights file.
    device : str
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``, etc.).
    """
    return _load_model_cached(str(Path(weights_path).resolve()), str(device))


@lru_cache(maxsize=4)
def _load_model_cached(weights_path: str, device: str):
    torch = _require_torch()
    from ecgdatakit.processing.nn.deepfade import DeepFADE

    model = DeepFADE(**DeepFADE.DEFAULT_ARGS).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = OrderedDict(
        (_remap_state_dict_key(k), v) for k, v in state_dict.items()
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def denoise_deepfade(
    lead: LeadLike,
    weights_path: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
    *,
    fs: int | None = None,
) -> Lead:
    """Denoise an ECG lead using the DeepFADE neural network (experimental).

    The signal is resampled to 500 Hz if needed and cut into 10 s windows
    with 50 % overlap (the ends are mirror-padded).  Each window is
    z-scored, passed through the network, and the output is rescaled by a
    least-squares fit to the 0.5--40 Hz band of the same window, so the
    result keeps the input amplitude units.  Windows are recombined with a
    Hann overlap-add, resampled back to the original rate and trimmed to
    the original length.

    The output has baseline wander removed: its isoelectric level is near
    zero, not at the input mean, and the input offset has no effect.
    Inference on CPU is deterministic.  A :class:`UserWarning` is issued on
    every call (once torch is available) because the model is experimental.

    Caveats: the fitted gain is unbiased only when the network output does
    not correlate with the input noise, and the model is not
    sign-symmetric (inverted leads such as aVR give a lower correlation
    with the clean signal).

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead in physical units (for example mV), or a signal
        array.  Leads holding raw ADC counts are refused.
    weights_path : str | Path
        Path to the pre-trained ``.pt`` weights file.
    device : str
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).
    batch_size : int
        Inference batch size for multi-window signals.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    Lead
        Denoised lead with the same length, sample rate and metadata as the
        input (new object, original unchanged).

    Raises
    ------
    ValueError
        Raw-count lead, signal not 1-D, shorter than 1 s, containing NaN or
        infinite values, or non-positive *batch_size*.
    """
    torch = _require_torch()
    warnings.warn(
        "DeepFADE denoising is experimental: the model has not been "
        "validated on public benchmarks; review its output before use",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )
    if isinstance(lead, Lead) and _is_raw_counts(lead):
        raise ValueError(
            f"Lead {lead.label!r} holds raw ADC counts; convert it with "
            "lead.to_physical() or parse with auto_scale=True before denoising"
        )
    lead = ensure_lead(lead, fs=fs)
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    original_fs = lead.sampling_rate
    if original_fs is None or original_fs <= 0:
        raise ValueError(f"sampling rate must be positive, got {original_fs}")
    samples = np.asarray(lead.samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {samples.shape}")
    if samples.size < int(np.ceil(_MIN_DURATION_S * original_fs)):
        raise ValueError(
            f"signal too short: {samples.size} samples at {original_fs} Hz, "
            f"need at least {_MIN_DURATION_S:g} s"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("signal contains NaN or infinite values")

    sig = require_scipy("signal")
    model = _load_model(weights_path, device)

    # The output is referenced to its own isoelectric level, so the input
    # offset carries no information; removing it keeps resampling edge
    # effects independent of the offset.
    signal = _resample(samples - np.median(samples), original_fs, _EXPECTED_FS)
    n = len(signal)
    padded, pad_left = _pad(signal, _EXPECTED_LEN, _HOP)
    sos = sig.butter(2, [0.5, 40.0], btype="band", fs=_EXPECTED_FS, output="sos")
    reference = sig.sosfiltfilt(sos, padded)

    starts = list(range(0, len(padded) - _EXPECTED_LEN + 1, _HOP))
    outputs = np.zeros((len(starts), _EXPECTED_LEN))
    for b in range(0, len(starts), batch_size):
        chunk = starts[b : b + batch_size]
        segs = np.stack([padded[s : s + _EXPECTED_LEN] for s in chunk])
        mu = segs.mean(axis=1, keepdims=True)
        sd = segs.std(axis=1, keepdims=True)
        z = np.divide(segs - mu, sd, out=np.zeros_like(segs), where=sd > 0)
        x = torch.tensor(z[:, np.newaxis, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            clean, _baseline = model(x)
        out = clean.squeeze(1).cpu().numpy().astype(np.float64)
        for k, s in enumerate(chunk):
            outputs[b + k] = _rescale(out[k], reference[s : s + _EXPECTED_LEN])

    denoised = _overlap_add(outputs, starts, len(padded))[pad_left : pad_left + n]
    denoised = _resample(denoised, _EXPECTED_FS, original_fs)
    denoised = _fit_length(denoised, samples.size)
    return new_lead(lead, samples=denoised)


def _is_raw_counts(lead: Lead) -> bool:
    """True when a Lead carries scaling metadata and is still unscaled."""
    if not lead.is_raw:
        return False
    return bool(
        lead.resolution != 1.0
        or lead.offset != 0.0
        or lead.resolution_unit
        or lead.adc_resolution
    )


def _resample(x: NDArray[np.float64], fs_in: int, fs_out: int) -> NDArray[np.float64]:
    if fs_in == fs_out:
        return x
    from math import gcd

    sig = require_scipy("signal")
    g = gcd(int(fs_in), int(fs_out))
    return sig.resample_poly(
        x, int(fs_out) // g, int(fs_in) // g, padtype="line"
    ).astype(np.float64)


def _fit_length(x: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)), mode="edge")


def _pad(x: NDArray[np.float64], seg_len: int, hop: int) -> tuple[NDArray[np.float64], int]:
    """Mirror-pad so every original sample is covered by full overlap.

    Returns the padded signal and the left padding length.  The total length
    is ``seg_len + k * hop`` for some integer ``k >= 0``.
    """
    left = hop
    excess = len(x) + 2 * hop - seg_len
    right = hop + (-excess) % hop if excess > 0 else seg_len - len(x) - left
    return np.pad(x, (left, right), mode="reflect"), left


def _rescale(out: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale the network output to the input units.

    The output is referenced to its median (isoelectric level) and scaled
    by the least-squares gain onto the 0.5--40 Hz band of the input
    window.  Noise in the input is uncorrelated with the output, so the
    gain is not inflated by it.
    """
    centred = out - np.median(out)
    oc = centred - centred.mean()
    denom = float(np.dot(oc, oc))
    if denom <= 0:
        return np.zeros_like(out)
    gain = float(np.dot(oc, reference - reference.mean())) / denom
    return gain * centred


def _overlap_add(segments: NDArray[np.float64], starts: list[int], length: int) -> NDArray[np.float64]:
    """Recombine overlapping windows with a periodic Hann taper."""
    seg_len = segments.shape[1]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(seg_len) / seg_len)
    acc = np.zeros(length)
    weight = np.zeros(length)
    for seg, s in zip(segments, starts):
        acc[s : s + seg_len] += window * seg
        weight[s : s + seg_len] += window
    return np.divide(acc, weight, out=np.zeros(length), where=weight > 1e-8)
