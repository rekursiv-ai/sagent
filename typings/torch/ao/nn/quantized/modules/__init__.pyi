from typing import Any

from torch import Tensor
from torch.nn.modules.pooling import MaxPool2d as MaxPool2d

import torch
import torch.ao.nn.quantizable

from .activation import (
    ELU as ELU,
    Hardswish as Hardswish,
    LeakyReLU as LeakyReLU,
    MultiheadAttention as MultiheadAttention,
    PReLU as PReLU,
    ReLU6 as ReLU6,
    Sigmoid as Sigmoid,
    Softmax as Softmax,
)
from .batchnorm import (
    BatchNorm2d as BatchNorm2d,
    BatchNorm3d as BatchNorm3d,
)
from .conv import (
    Conv1d as Conv1d,
    Conv2d as Conv2d,
    Conv3d as Conv3d,
    ConvTranspose1d as ConvTranspose1d,
    ConvTranspose2d as ConvTranspose2d,
    ConvTranspose3d as ConvTranspose3d,
)
from .dropout import Dropout as Dropout
from .embedding_ops import (
    Embedding as Embedding,
    EmbeddingBag as EmbeddingBag,
)
from .functional_modules import (
    FloatFunctional as FloatFunctional,
    FXFloatFunctional as FXFloatFunctional,
    QFunctional as QFunctional,
)
from .linear import Linear as Linear
from .normalization import (
    GroupNorm as GroupNorm,
    InstanceNorm1d as InstanceNorm1d,
    InstanceNorm2d as InstanceNorm2d,
    InstanceNorm3d as InstanceNorm3d,
    LayerNorm as LayerNorm,
)
from .rnn import LSTM as LSTM

__all__ = [
    "ELU",
    "LSTM",
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "Dropout",
    "Embedding",
    "EmbeddingBag",
    "FXFloatFunctional",
    "FloatFunctional",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "MultiheadAttention",
    "PReLU",
    "QFunctional",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
]

class Quantize(torch.nn.Module):
    scale: torch.Tensor
    zero_point: torch.Tensor
    def __init__(self, scale, zero_point, dtype, factory_kwargs=...) -> None: ...
    def forward(self, X) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=...) -> Quantize: ...
    def extra_repr(self) -> str: ...

class DeQuantize(torch.nn.Module):
    def forward(self, Xq): ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=...) -> DeQuantize: ...
