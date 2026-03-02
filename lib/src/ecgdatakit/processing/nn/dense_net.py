"""DenseNet building blocks for 1-D signals.

Classes
-------
DenseBlock
    Dense connectivity: each layer receives all preceding feature maps.
TransitionBlock
    Dimension reduction / expansion between dense blocks.
DenseTrunk
    Full DenseNet backbone composed of DenseBlocks and TransitionBlocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import Activation, Conv1DBlock


class TransitionBlock(nn.Module):
    """Transition layer between dense blocks.

    Performs optional compression (1x1 conv) followed by pooling
    (avg, max, strided conv) or upsampling (ConvTranspose1d).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float | None = None,
        compression: float = 1.0,
        activation: dict | None = None,
        padding: int = 0,
        strides: int = 2,
        dilation: int = 1,
        pool_type: str = "avg",
        pool_size: int = 2,
        convolution_kernel: int = 3,
        transposition_channels: int = 1,
        transposition_kernel: int = 3,
        enable_compression_block: bool = True,
    ) -> None:
        super().__init__()

        self.pool_type = pool_type
        self.enable_compression_block = enable_compression_block
        self.dropout_rate = dropout_rate

        if activation is None:
            activation = {"name": "leaky_relu", "args": {"negative_slope": 0.1}}

        if enable_compression_block:
            self.compression_block = Conv1DBlock(
                in_channels=in_channels,
                out_channels=int(out_channels * compression),
                kernel=1,
                activation=activation,
                padding=padding,
                order="conv_bn_act",
            )
            self._output_channels = int(out_channels * compression)
            in_channels = self._output_channels
        else:
            self._output_channels = out_channels

        if dropout_rate:
            self.dropout_layer = nn.Dropout1d(p=dropout_rate)

        if pool_type == "avg":
            self.pooling_layer = nn.AvgPool1d(kernel_size=pool_size, stride=strides)
        elif pool_type == "max":
            self.pooling_layer = nn.MaxPool1d(kernel_size=pool_size, stride=strides)
        elif pool_type == "convolution":
            cp = convolution_kernel // 2
            self.pooling_layer = Conv1DBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel=convolution_kernel,
                activation=activation,
                padding=cp,
                strides=strides,
                order="conv_bn_act",
            )
        elif pool_type == "conv_transpose":
            tp = transposition_kernel // 2
            transposition_channels = 1
            self.deconvolution_layer = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels,
                    transposition_channels,
                    kernel_size=transposition_kernel,
                    padding=tp,
                    stride=strides,
                    output_padding=strides - 1,
                ),
                nn.BatchNorm1d(transposition_channels),
                Activation(**activation),
            )
            self._output_channels = transposition_channels
        else:
            raise ValueError(f"Invalid pool_type {pool_type!r}")

    def get_output_channels(self) -> int:
        return self._output_channels

    def forward(self, x):
        logits = x
        if self.enable_compression_block:
            logits = self.compression_block(logits)
        if self.dropout_rate:
            logits = self.dropout_layer(logits)
        if self.pool_type in ("avg", "max", "convolution"):
            logits = self.pooling_layer(logits)
        elif self.pool_type == "conv_transpose":
            logits = self.deconvolution_layer(logits)
        return logits


class DenseBlock(nn.Module):
    """Dense block with concatenated feature maps.

    Each internal layer receives all preceding feature maps as input
    (dense connectivity pattern from DenseNet).
    """

    def __init__(
        self,
        layers: int,
        in_channels: int,
        out_channels: int,
        kernel: int,
        growth_rate: int,
        dropout_rate: float | None = None,
        bottleneck: bool = False,
        activation: dict | None = None,
        grow_channels: bool = True,
    ) -> None:
        super().__init__()

        if activation is None:
            activation = {"name": "leaky_relu", "args": {"negative_slope": 0.1}}

        self._layer_list: list[Conv1DBlock] = []
        for i in range(layers):
            layer = Conv1DBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                dropout_rate=dropout_rate,
                bottleneck=bottleneck,
                order="conv_bn_act",
                activation=activation,
                padding=kernel // 2,
                strides=1,
                dilation=1,
            )
            self.add_module(f"DenseLayer_{i}", layer)
            self._layer_list.append(layer)
            in_channels += out_channels
            if grow_channels:
                out_channels += growth_rate

        self._output_channels = in_channels

    def get_output_channels(self) -> int:
        return self._output_channels

    def forward(self, x):
        logits = x
        feature_maps = [logits]
        for layer in self._layer_list:
            feature_maps.append(layer(logits))
            logits = torch.cat(feature_maps, dim=1)
        return logits


class DenseTrunk(nn.Module):
    """Complete DenseNet backbone: initial conv → (DenseBlock + Transition) * N.

    Parameters
    ----------
    input_channels : int
        Number of input channels (1 for single-lead ECG).
    blocks : int
        Number of dense blocks.
    layers : int | list[int]
        Layers per block (auto-calculated from *depth* if 0).
    transition_block_kwargs : list[dict] | None
        Per-transition-block keyword arguments (length = blocks - 1).
    """

    def __init__(
        self,
        input_channels: int,
        blocks: int = 3,
        layers: int | list[int] = 0,
        kernels: int | list[int] = 3,
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
        transition_block_kwargs: list[dict] | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        assert 0.0 < compression <= 1.0

        if isinstance(layers, list):
            assert len(layers) == blocks
        elif layers == 0:
            n = (depth - (blocks + 1)) // (blocks * (2 if bottleneck else 1))
            layers = [int(n)] * blocks
        else:
            layers = [int(layers)] * blocks

        if isinstance(kernels, (list, tuple)):
            assert len(kernels) == blocks
        else:
            kernels = [int(kernels)] * blocks

        use_transition_kwargs = False
        if transition_block_kwargs is not None:
            assert len(transition_block_kwargs) == blocks - 1
            use_transition_kwargs = True

        if activation is None:
            activation = {"name": "relu"}

        filters = growth_rate * 2
        initial_kernel = 3
        if conv_padding is None:
            conv_padding = initial_kernel // 2

        self._initial_layer = Conv1DBlock(
            in_channels=input_channels,
            out_channels=filters,
            kernel=initial_kernel,
            activation=activation,
            padding=conv_padding,
            strides=conv_strides,
            dilation=conv_dilation,
            order="conv_bn_act",
        )

        self._blocks: nn.ModuleList = nn.ModuleList()
        future_in_channels = filters
        for b in range(blocks):
            block = DenseBlock(
                layers=layers[b],
                in_channels=future_in_channels,
                out_channels=filters,
                kernel=kernels[b],
                growth_rate=growth_rate,
                grow_channels=grow_layers_channels,
                bottleneck=bottleneck,
                activation=activation,
            )
            self._blocks.append(block)
            future_in_channels = block.get_output_channels()
            filters = growth_rate * layers[b]

            if b < blocks - 1:
                tkw = transition_block_kwargs[b] if use_transition_kwargs else {}
                transition = TransitionBlock(
                    in_channels=future_in_channels,
                    out_channels=filters,
                    dropout_rate=dropout_rate,
                    compression=compression,
                    activation=activation,
                    transposition_channels=filters,
                    **tkw,
                )
                self._blocks.append(transition)
                future_in_channels = transition.get_output_channels()
                filters = int(filters * compression)

        self._output_channels = future_in_channels

    def get_output_channels(self) -> int:
        return self._output_channels

    def forward(self, x):
        logits = self._initial_layer(x)
        for block in self._blocks:
            logits = block(logits)
        return logits
