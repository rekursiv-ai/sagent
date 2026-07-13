from torch.ao.nn.quantized.modules import (
    DeQuantize as DeQuantize,
    MaxPool2d as MaxPool2d,
    Quantize as Quantize,
    activation as activation,
    batchnorm as batchnorm,
    conv as conv,
    dropout as dropout,
    embedding_ops as embedding_ops,
    functional_modules as functional_modules,
    linear as linear,
    normalization as normalization,
    rnn as rnn,
    utils as utils,
)
from torch.ao.nn.quantized.modules.activation import (
    ELU as ELU,
    Hardswish as Hardswish,
    LeakyReLU as LeakyReLU,
    MultiheadAttention as MultiheadAttention,
    PReLU as PReLU,
    ReLU6 as ReLU6,
    Sigmoid as Sigmoid,
    Softmax as Softmax,
)
from torch.ao.nn.quantized.modules.batchnorm import (
    BatchNorm2d as BatchNorm2d,
    BatchNorm3d as BatchNorm3d,
)
from torch.ao.nn.quantized.modules.conv import (
    Conv1d as Conv1d,
    Conv2d as Conv2d,
    Conv3d as Conv3d,
    ConvTranspose1d as ConvTranspose1d,
    ConvTranspose2d as ConvTranspose2d,
    ConvTranspose3d as ConvTranspose3d,
)
from torch.ao.nn.quantized.modules.dropout import Dropout as Dropout
from torch.ao.nn.quantized.modules.embedding_ops import (
    Embedding as Embedding,
    EmbeddingBag as EmbeddingBag,
)
from torch.ao.nn.quantized.modules.functional_modules import (
    FloatFunctional as FloatFunctional,
    FXFloatFunctional as FXFloatFunctional,
    QFunctional as QFunctional,
)
from torch.ao.nn.quantized.modules.linear import Linear as Linear
from torch.ao.nn.quantized.modules.normalization import (
    GroupNorm as GroupNorm,
    InstanceNorm1d as InstanceNorm1d,
    InstanceNorm2d as InstanceNorm2d,
    InstanceNorm3d as InstanceNorm3d,
    LayerNorm as LayerNorm,
)
from torch.ao.nn.quantized.modules.rnn import LSTM as LSTM

r"""Quantized Modules.

Note::
    The `torch.nn.quantized` namespace is in the process of being deprecated.
    Please, use `torch.ao.nn.quantized` instead.
"""
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
