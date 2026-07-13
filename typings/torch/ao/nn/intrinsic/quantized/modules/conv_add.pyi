import torch.ao.nn.quantized as nnq

_reverse_repeat_padding = ...

class ConvAdd2d(nnq.Conv2d):
    _FLOAT_MODULE = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input, extra_input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point) -> Self: ...

class ConvAddReLU2d(nnq.Conv2d):
    _FLOAT_MODULE = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input, extra_input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point) -> Self: ...
