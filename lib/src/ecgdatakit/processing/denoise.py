"""ECG denoising using the DeepFADE neural network.

DeepFADE is a DenseNet encoder-decoder that removes noise and baseline
wander from single-lead ECG signals.  Operates on 10-second segments
at 500 Hz (5 000 samples).

Requires: ``pip install ecgdatakit[denoising]`` (torch >= 2.0)
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead
from ecgdatakit.processing._core import new_lead

if TYPE_CHECKING:
    import torch

_EXPECTED_FS = 500
_EXPECTED_LEN = 5000


def _require_torch():
    """Lazily import torch, raising a helpful error if missing."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for DeepFADE denoising. "
            "Install it with: pip install ecgdatakit[denoising]"
        ) from exc


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
    """Load a DeepFADE model with pre-trained weights.

    Parameters
    ----------
    weights_path : str | Path
        Path to the ``.pt`` weights file.
    device : str
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``, etc.).
    """
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
    lead: Lead,
    weights_path: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
) -> Lead:
    """Denoise an ECG lead using the DeepFADE neural network.

    Automatically resamples to 500 Hz if needed, segments into
    5 000-sample chunks, runs inference, and reassembles.

    Parameters
    ----------
    lead : Lead
        Input ECG lead (single channel).
    weights_path : str | Path
        Path to the pre-trained ``.pt`` weights file.
    device : str
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).
    batch_size : int
        Inference batch size for multi-segment signals.

    Returns
    -------
    Lead
        Denoised lead (new object, original unchanged).
    """
    torch = _require_torch()

    original_fs = lead.sample_rate
    if original_fs != _EXPECTED_FS:
        from ecgdatakit.processing.resample import resample
        lead_500 = resample(lead, _EXPECTED_FS)
    else:
        lead_500 = lead

    signal = lead_500.samples.astype(np.float64)
    segments = _segment(signal, _EXPECTED_LEN)
    model = _load_model(weights_path, device)

    denoised_segments: list[NDArray] = []
    for i in range(0, len(segments), batch_size):
        batch = np.stack(segments[i : i + batch_size])
        batch_tensor = torch.tensor(
            batch[:, np.newaxis, :], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            clean_signal, _baseline = model(batch_tensor)
        denoised_segments.append(clean_signal.squeeze(1).cpu().numpy())

    all_segments = np.concatenate(denoised_segments, axis=0)
    denoised = _reassemble(all_segments, len(signal), _EXPECTED_LEN)
    result = new_lead(lead_500, samples=denoised.astype(np.float64))

    if original_fs != _EXPECTED_FS:
        from ecgdatakit.processing.resample import resample
        result = resample(result, original_fs)

    return result


def _segment(signal: NDArray, seg_len: int) -> list[NDArray]:
    """Split signal into fixed-length segments, zero-padding the last."""
    n = len(signal)
    segments = []
    for start in range(0, n, seg_len):
        chunk = signal[start : start + seg_len]
        if len(chunk) < seg_len:
            padded = np.zeros(seg_len, dtype=np.float64)
            padded[: len(chunk)] = chunk
            chunk = padded
        segments.append(chunk)
    return segments


def _reassemble(segments: NDArray, original_len: int, seg_len: int) -> NDArray:
    """Concatenate segments and trim to original length."""
    return segments.reshape(-1)[:original_len]
