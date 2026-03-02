"""Fundamental neural-network building blocks used by DeepFADE.

Provides :class:`Activation` (a generic wrapper around PyTorch activations)
and :class:`Conv1DBlock` (1-D convolution + batch-norm + activation with
configurable ordering).
"""

from __future__ import annotations

import torch.nn as nn


_ACTIVATIONS = {
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "softmax": nn.Softmax,
    "sigmoid": nn.Sigmoid,
}


class Activation(nn.Module):
    """Configurable activation function.

    Parameters
    ----------
    desc : dict | None
        ``{"name": "elu", "args": {"alpha": 0.1}}``.  Takes priority.
    name : str | None
        Activation name when *desc* is not given.
    args : dict | list | None
        Arguments forwarded to the PyTorch activation constructor.
    """

    def __init__(
        self,
        desc: dict | None = None,
        name: str | None = None,
        args: dict | list | tuple | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if desc is not None:
            _name = desc["name"]
            _args = desc.get("args")
        else:
            _name = name
            _args = args

        if _name not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {_name!r}; choose from {list(_ACTIVATIONS)}"
            )

        cls = _ACTIVATIONS[_name]
        if isinstance(_args, (list, tuple)):
            self._act = cls(*_args)
        elif isinstance(_args, dict):
            self._act = cls(**_args)
        elif _args is None:
            self._act = cls()
        else:
            raise TypeError(f"Unexpected args type: {type(_args)}")

    def forward(self, x):
        return self._act(x)



class Conv1DBlock(nn.Module):
    """1-D convolution block with configurable ordering.

    Supported *order* values:

    - ``"bn_act_conv"`` — BatchNorm → Activation → Conv (pre-activation)
    - ``"conv_bn_act"`` — Conv → BatchNorm → Activation (post-activation)
    - ``"conv_act"``    — Conv → Activation (no BatchNorm)
    - ``"bn_act"``      — BatchNorm → Activation (no convolution)

    Parameters
    ----------
    in_channels, out_channels : int
        Channel counts.
    kernel : int
        Convolution kernel size.
    dropout_rate : float | None
        Dropout probability (``None`` = no dropout).
    bottleneck : bool | str
        ``False``, ``"before"``, or ``"after"``.
    activation : dict | None
        Activation descriptor; defaults to LeakyReLU(0.1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        dropout_rate: float | None = None,
        bottleneck: bool | str = False,
        bottleneck_width: int = 4,
        activation: dict | None = None,
        dilation: int = 1,
        padding: int = 0,
        strides: int = 1,
        order: str = "bn_act_conv",
    ) -> None:
        super().__init__()

        self.bottleneck = bottleneck
        if activation is None:
            activation = {"name": "leaky_relu", "args": {"negative_slope": 0.1}}

        if bottleneck:
            bn_in = in_channels if bottleneck == "before" else out_channels
            if bottleneck == "before":
                in_channels = out_channels * bottleneck_width

            if order == "bn_act_conv":
                self._bottleneck_layers = nn.Sequential(
                    nn.BatchNorm1d(in_channels),
                    Activation(**activation),
                    nn.Conv1d(in_channels, out_channels * bottleneck_width,
                              kernel_size=1, bias=True, stride=strides),
                )
            else:
                self._bottleneck_layers = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels * bottleneck_width,
                              kernel_size=1, bias=True, stride=strides),
                    nn.BatchNorm1d(out_channels * bottleneck_width),
                    Activation(**activation),
                )
            if bottleneck == "after":
                extra: list[nn.Module]
                if order == "bn_act_conv":
                    extra = [
                        nn.BatchNorm1d(bn_in),
                        Activation(**activation),
                        nn.Conv1d(bn_in, out_channels, kernel_size=kernel,
                                  bias=True, stride=strides),
                    ]
                else:
                    extra = [
                        nn.Conv1d(bn_in, out_channels, kernel_size=kernel,
                                  bias=True, stride=strides),
                        nn.BatchNorm1d(out_channels),
                        Activation(**activation),
                    ]
                for m in extra:
                    self._bottleneck_layers.append(m)
            if dropout_rate:
                self._bottleneck_layers.append(nn.Dropout1d(p=dropout_rate))

        if order == "bn_act_conv":
            self._layers = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                Activation(**activation),
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel,
                          padding=padding, bias=True, stride=strides,
                          dilation=dilation),
            )
        elif order == "conv_bn_act":
            self._layers = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel,
                          padding=padding, bias=True, stride=strides,
                          dilation=dilation),
                nn.BatchNorm1d(out_channels),
                Activation(**activation),
            )
        elif order == "conv_act":
            self._layers = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel,
                          padding=padding, bias=True, stride=strides,
                          dilation=dilation),
                Activation(**activation),
            )
        elif order == "bn_act":
            self._layers = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                Activation(**activation),
            )
        else:
            raise ValueError(f"Unknown order {order!r}")

        if dropout_rate:
            self._layers.append(nn.Dropout1d(p=dropout_rate))

    def forward(self, x):
        y = x
        if self.bottleneck == "before":
            y = self._bottleneck_layers(x)
        y = self._layers(y)
        if self.bottleneck == "after":
            y = self._bottleneck_layers(x)
        return y
