import types

from .modules import *
from .modules.fused import _FusedModule as _FusedModule

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

def __getattr__(name: str) -> types.ModuleType: ...
