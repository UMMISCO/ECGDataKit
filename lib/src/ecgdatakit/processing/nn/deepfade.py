"""DeepFADE — Deep Filtering and Artefact Denoising for ECG.

An encoder-decoder neural network built on DenseNet trunks that removes
noise and baseline wander from single-lead ECG signals.  The forward pass
returns ``(clean_signal, estimated_baseline)``.

Architecture
~~~~~~~~~~~~

::

    Input (batch, 1, 5000)
         │
     ┌───▼────┐
     │Encoder │  DenseTrunk(pool_steps=[2,2,2,5]) → 8-ch latent
     └───┬────┘
         │
     ┌───▼────┐
     │Decoder │  DenseTrunk(up_steps=[5,2,2,2]) → signal + baseline heads
     └───┬────┘
         │
    (signal_out, baseline_out)   each (batch, 1, 5000)
"""

from __future__ import annotations

import torch.nn as nn

from .dense_net import DenseTrunk
from .layers import Conv1DBlock


class DenseEncoder(nn.Module):
    """Encoding path: DenseTrunk + latent-space projection."""

    def __init__(
        self,
        input_channels: int = 1,
        layers: int = 0,
        kernels: int = 3,
        growth_rate: int = 12,
        dropout_rate: float | None = None,
        bottleneck: bool = False,
        compression: float = 1.0,
        depth: int = 40,
        activation: dict | None = None,
        conv_padding: int | None = None,
        conv_strides: int = 1,
        conv_dilation: int = 1,
        grow_layers_channels: bool = True,
        pool_steps: list[int] | None = None,
        verbose: bool = False,
        pool_type: str = "avg",
    ) -> None:
        super().__init__()
        if pool_steps is None:
            pool_steps = [2]
        if activation is None:
            activation = {"name": "relu"}

        transition_kwargs = [
            {"pool_type": pool_type, "pool_size": s, "strides": s}
            for s in pool_steps
        ]
        blocks = len(transition_kwargs) + 1

        self._trunk = DenseTrunk(
            input_channels=input_channels,
            blocks=blocks,
            layers=layers,
            kernels=kernels,
            growth_rate=growth_rate,
            dropout_rate=dropout_rate,
            bottleneck=bottleneck,
            compression=compression,
            depth=depth,
            activation=activation,
            conv_padding=conv_padding,
            conv_strides=conv_strides,
            conv_dilation=conv_dilation,
            grow_layers_channels=grow_layers_channels,
            transition_block_kwargs=transition_kwargs,
            verbose=verbose,
        )

        latent_channels = 8
        self._latent_conv = Conv1DBlock(
            in_channels=self._trunk.get_output_channels(),
            out_channels=latent_channels,
            kernel=3,
            activation=activation,
            padding=1,
            strides=1,
            dilation=conv_dilation,
            order="conv_bn_act",
        )
        self._output_channels = latent_channels

    def get_output_channels(self) -> int:
        return self._output_channels

    def forward(self, x):
        return self._latent_conv(self._trunk(x))


class DenseDecoder(nn.Module):
    """Decoding path: DenseTrunk + dual output heads (signal + baseline)."""

    def __init__(
        self,
        input_channels: int = 1,
        layers: int = 0,
        kernels: int = 3,
        growth_rate: int = 12,
        dropout_rate: float | None = None,
        bottleneck: bool = False,
        compression: float = 1.0,
        depth: int = 40,
        activation: dict | None = None,
        conv_padding: int | None = None,
        conv_strides: int = 1,
        conv_dilation: int = 1,
        grow_layers_channels: bool = True,
        up_steps: list[int] | None = None,
        verbose: bool = False,
        enable_final_activation: bool = False,
    ) -> None:
        super().__init__()
        if up_steps is None:
            up_steps = [2]
        if activation is None:
            activation = {"name": "relu"}

        transition_kwargs = [
            {"pool_type": "conv_transpose", "strides": s}
            for s in up_steps
        ]
        blocks = len(transition_kwargs) + 1

        self._trunk = DenseTrunk(
            input_channels=input_channels,
            blocks=blocks,
            layers=layers,
            kernels=kernels,
            growth_rate=growth_rate,
            dropout_rate=dropout_rate,
            bottleneck=bottleneck,
            compression=compression,
            depth=depth,
            activation=activation,
            conv_padding=conv_padding,
            conv_strides=conv_strides,
            conv_dilation=conv_dilation,
            grow_layers_channels=grow_layers_channels,
            transition_block_kwargs=transition_kwargs,
            verbose=verbose,
        )

        out_ch = self._trunk.get_output_channels()
        act = {"name": "elu", "args": {}}

        self._signal_output = nn.Sequential(
            Conv1DBlock(out_ch, 32, kernel=3, activation=act, padding=1,
                        strides=1, dilation=conv_dilation, order="conv_bn_act"),
            Conv1DBlock(32, 16, kernel=3, activation=act, padding=1,
                        strides=1, dilation=conv_dilation, order="conv_bn_act"),
            nn.Conv1d(16, 1, kernel_size=3, padding=1, bias=True,
                      stride=1, dilation=conv_dilation),
        )
        self._baseline_output = nn.Sequential(
            Conv1DBlock(out_ch, 32, kernel=3, activation=act, padding=1,
                        strides=1, dilation=conv_dilation, order="conv_bn_act"),
            Conv1DBlock(32, 16, kernel=3, activation=act, padding=1,
                        strides=1, dilation=conv_dilation, order="conv_bn_act"),
            nn.Conv1d(16, 1, kernel_size=3, padding=1, bias=True,
                      stride=1, dilation=conv_dilation),
        )

    def forward(self, x):
        logits = self._trunk(x)
        return self._signal_output(logits), self._baseline_output(logits)


class DeepFADE(nn.Module):
    """DeepFADE encoder-decoder for ECG denoising.

    Parameters
    ----------
    pool_steps : list[int]
        Encoder downsampling factors (default ``[2, 2, 2, 5]``).
    up_steps : list[int]
        Decoder upsampling factors (default ``[5, 2, 2, 2]``).
    layers : int
        Dense layers per block.
    compression : float
        Channel compression ratio at transitions (0–1, 1 = no compression).
    activation : dict | None
        Activation descriptor, e.g. ``{"name": "elu", "args": {"alpha": 0.1}}``.
    encoder_pool_type : str
        Pooling strategy for encoder (``"avg"``, ``"max"``, ``"convolution"``).

    Forward
    -------
    Input:  ``(batch, 1, signal_length)``
    Output: ``(clean_signal, estimated_baseline)`` each ``(batch, 1, signal_length)``
    """

    DEFAULT_ARGS: dict = {
        "verbose": False,
        "encoder_pool_type": "convolution",
        "enable_final_activation": False,
        "layers": 8,
        "compression": 1,
        "bottleneck": False,
        "pool_steps": [2, 2, 2, 5],
        "up_steps": [5, 2, 2, 2],
        "activation": {"name": "elu", "args": {"alpha": 0.1}},
        "dropout_rate": 0.2,
    }

    def __init__(
        self,
        layers: int = 0,
        kernels: int = 3,
        growth_rate: int = 12,
        dropout_rate: float | None = None,
        bottleneck: bool = False,
        compression: float = 1.0,
        depth: int = 40,
        activation: dict | None = None,
        conv_padding: int | None = None,
        conv_strides: int = 1,
        conv_dilation: int = 1,
        grow_layers_channels: bool = True,
        pool_steps: list[int] | None = None,
        up_steps: list[int] | None = None,
        verbose: bool = False,
        enable_final_activation: bool = False,
        encoder_pool_type: str = "avg",
    ) -> None:
        super().__init__()
        if pool_steps is None:
            pool_steps = [2]
        if up_steps is None:
            up_steps = [2]

        common = dict(
            layers=layers, kernels=kernels, growth_rate=growth_rate,
            dropout_rate=dropout_rate, bottleneck=bottleneck,
            compression=compression, depth=depth, activation=activation,
            conv_padding=conv_padding, conv_strides=conv_strides,
            conv_dilation=conv_dilation,
            grow_layers_channels=grow_layers_channels, verbose=verbose,
        )

        self._encoder = DenseEncoder(
            input_channels=1, pool_type=encoder_pool_type,
            pool_steps=pool_steps, **common,
        )
        self._decoder = DenseDecoder(
            input_channels=self._encoder.get_output_channels(),
            up_steps=up_steps,
            enable_final_activation=enable_final_activation,
            **common,
        )

    def forward(self, x):
        encoded = self._encoder(x)
        return self._decoder(encoded)
