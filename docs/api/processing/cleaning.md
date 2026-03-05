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
| `"default"` | scipy | Bandpass 0.5–40 Hz + 50 Hz notch |
| `"biosppy"` | `pip install biosppy` | BioSPPy ECG filter |
| `"neurokit2"` | `pip install neurokit2` | NeuroKit2 adaptive pipeline |
| `"combined"` | biosppy + neurokit2 | BioSPPy → NeuroKit2 |
| `"deepfade"` | `pip install torch` | DeepFADE denoising autoencoder |

## DeepFADE

DeepFADE is a denoising autoencoder developed as part of ECGDataKit, trained on a large private multi-source ECG database with extensive noise augmentations (baseline wander, electrode motion, muscle artifacts, powerline interference). The architecture follows a symmetric DenseNet encoder-decoder design: the encoder compresses a 10-second single-lead ECG segment (500 Hz, 5 000 samples) through four dense blocks with progressive downsampling into an 8-channel latent representation, while the decoder mirrors the path with transposed-convolution upsampling and produces two outputs — the denoised signal and the estimated baseline wander. Pre-trained weights are bundled with the package.

```python
from ecgdatakit.processing import clean_ecg

# CPU inference (default)
denoised = clean_ecg(lead, method="deepfade")

# GPU acceleration
denoised = clean_ecg(lead, method="deepfade", device="cuda")

# Apple Silicon MPS
denoised = clean_ecg(lead, method="deepfade", device="mps")

# Custom weights and batch size
denoised = clean_ecg(lead, method="deepfade", weights_path="my_weights.pt", batch_size=64)
```

```{tip}
Signals are automatically resampled to 500 Hz, segmented into 5 000-sample chunks, denoised in batches, and reassembled to the original length and sample rate.
```
