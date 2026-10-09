# ECG Cleaning

Unified cleaning interface with multiple backends.

| | |
|---|---|
| {func}`~ecgdatakit.processing.clean_ecg` | Clean an ECG lead signal |

```{eval-rst}
.. currentmodule:: ecgdatakit.processing

.. autofunction:: clean_ecg
```

## Available methods

| Method | Extra dependency | Description |
|--------|-----------------|-------------|
| `"default"` | scipy | Bandpass 0.5–40 Hz + power-line notch |
| `"biosppy"` | `pip install biosppy` | BioSPPy ECG filter (no power-line stage) |
| `"neurokit2"` | `pip install neurokit2` | NeuroKit2 `ecg_clean` (0.5 Hz high-pass + power-line filter) |
| `"combined"` | biosppy + neurokit2 | BioSPPy → NeuroKit2 |
| `"deepfade"` | `pip install "ecgdatakit[denoising]"` (torch) | DeepFADE denoising autoencoder (experimental) |

Below 89 Hz the 40 Hz upper cutoff of `"default"` is lowered to 0.45 * fs, with a warning.

`powerline` sets the mains frequency (default 50 Hz, use `powerline=60` in the Americas) for `"default"`, `"neurokit2"` and the NeuroKit2 stage of `"combined"`. `powerline=None` disables the notch of `"default"`. When the sampling rate is too low for the notch, `"default"` skips it with a warning.

Backend errors are never hidden: a signal a backend cannot process raises its error, and NaN or Inf samples raise `ValueError`. Keyword arguments that the selected method does not use raise `TypeError`; `device`, `weights_path` and `batch_size` are accepted by `"deepfade"` only.

```python
cleaned = clean_ecg(lead)                    # 0.5-40 Hz + 50 Hz notch
cleaned = clean_ecg(lead, powerline=60)      # 60 Hz mains
cleaned = clean_ecg(lead, method="neurokit2", powerline=60)
```

## DeepFADE (experimental)

```{warning}
DeepFADE is experimental. The model has not been validated on public benchmarks, and every call issues a `UserWarning`. Review its output before using it for analysis.
```

DeepFADE is a denoising autoencoder developed as part of ECGDataKit, trained on a large private multi-source ECG database with noise augmentations (baseline wander, electrode motion, muscle artifacts, powerline interference). It is a symmetric DenseNet encoder-decoder: the encoder compresses a 10-second single-lead segment (500 Hz, 5 000 samples) through four dense blocks into an 8-channel latent representation, and the decoder mirrors the path with transposed-convolution upsampling. Pre-trained weights are bundled with the package. Requires `pip install "ecgdatakit[denoising]"` (torch >= 2.0).

How a lead is processed:

1. The signal is resampled to 500 Hz if needed and cut into 10-second windows with 50 % overlap. The ends are mirror-padded.
2. Each window is z-scored and passed through the network.
3. The network output has no fixed amplitude scale. It is rescaled by a least-squares fit to the 0.5 to 40 Hz band of the same input window, so the result stays in the input units. The output is exactly proportional to the input amplitude, and adding a constant offset to the input does not change it.
4. Windows are recombined with a Hann overlap-add, resampled back to the original rate and trimmed to the original length.

The output has baseline wander removed: its isoelectric level is near zero, not at the input mean. CPU inference is deterministic. The lead must be in physical units (for example mV): a lead still holding raw ADC counts raises `ValueError`, so call `lead.to_physical()` or parse with `auto_scale=True` first. Signals shorter than 1 s, or containing NaN or infinite values, also raise `ValueError`. The experimental warning is issued only when torch is available; without torch the call raises `ImportError`.

Caveats:

- The gain in step 3 is unbiased only when the network output does not correlate with the noise in the input. Noise that the network partly reproduces (for example structured artefacts that resemble QRS complexes) inflates the gain.
- The model is not sign-symmetric: an inverted lead (for example aVR) is not simply the inverted result of the upright lead, and its output correlates less well with the clean signal.

```python
from ecgdatakit.processing import clean_ecg

denoised = clean_ecg(lead, method="deepfade")

# GPU (CUDA) or Apple Silicon (MPS)
denoised = clean_ecg(lead, method="deepfade", device="cuda")
denoised = clean_ecg(lead, method="deepfade", device="mps")

# Custom weights and batch size
denoised = clean_ecg(lead, method="deepfade", weights_path="my_weights.pt", batch_size=64)
```
