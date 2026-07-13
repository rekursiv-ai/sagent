import torch

__all__ = [
    "BNReLU2d",
    "BNReLU3d",
    "ConvAdd2d",
    "ConvAddReLU2d",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearBn1d",
    "LinearLeakyReLU",
    "LinearReLU",
    "LinearTanh",
]

class _FusedModule(torch.nn.Sequential): ...

class ConvReLU1d(_FusedModule):
    def __init__(self, conv, relu) -> None: ...

class ConvReLU2d(_FusedModule):
    def __init__(self, conv, relu) -> None: ...

class ConvReLU3d(_FusedModule):
    def __init__(self, conv, relu) -> None: ...

class LinearReLU(_FusedModule):
    def __init__(self, linear, relu) -> None: ...

class ConvBn1d(_FusedModule):
    def __init__(self, conv, bn) -> None: ...

class ConvBn2d(_FusedModule):
    def __init__(self, conv, bn) -> None: ...

class ConvBnReLU1d(_FusedModule):
    def __init__(self, conv, bn, relu) -> None: ...

class ConvBnReLU2d(_FusedModule):
    def __init__(self, conv, bn, relu) -> None: ...

class ConvBn3d(_FusedModule):
    def __init__(self, conv, bn) -> None: ...

class ConvBnReLU3d(_FusedModule):
    def __init__(self, conv, bn, relu) -> None: ...

class BNReLU2d(_FusedModule):
    def __init__(self, batch_norm, relu) -> None: ...

class BNReLU3d(_FusedModule):
    def __init__(self, batch_norm, relu) -> None: ...

class LinearBn1d(_FusedModule):
    def __init__(self, linear, bn) -> None: ...

class LinearLeakyReLU(_FusedModule):
    def __init__(self, linear, leaky_relu) -> None: ...

class LinearTanh(_FusedModule):
    def __init__(self, linear, tanh) -> None: ...

class ConvAdd2d(_FusedModule):
    def __init__(self, conv, add) -> None: ...
    def forward(self, x1, x2): ...

class ConvAddReLU2d(_FusedModule):
    def __init__(self, conv, add, relu) -> None: ...
    def forward(self, x1, x2): ...
