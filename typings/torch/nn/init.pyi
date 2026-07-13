from typing import (
    Literal,
    TypeVar,
)
from typing_extensions import ParamSpec

from torch import Tensor

import torch

"""This file contains utilities for initializing neural network parameters."""
__all__ = [
    "calculate_gain",
    "constant",
    "constant_",
    "dirac",
    "dirac_",
    "eye",
    "eye_",
    "kaiming_normal",
    "kaiming_normal_",
    "kaiming_uniform",
    "kaiming_uniform_",
    "normal",
    "normal_",
    "ones_",
    "orthogonal",
    "orthogonal_",
    "sparse",
    "sparse_",
    "trunc_normal_",
    "uniform",
    "uniform_",
    "xavier_normal",
    "xavier_normal_",
    "xavier_uniform",
    "xavier_uniform_",
    "zeros_",
]
_R = TypeVar("_R")
_P = ParamSpec("_P")
type _NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]
type _FanMode = Literal["fan_in", "fan_out"]

def calculate_gain(
    nonlinearity: _NonlinearityType, param: float | None = ...
) -> float: ...
def uniform_(
    tensor: Tensor,
    a: float = ...,
    b: float = ...,
    generator: torch.Generator | None = ...,
) -> Tensor: ...
def normal_(
    tensor: Tensor,
    mean: float = ...,
    std: float = ...,
    generator: torch.Generator | None = ...,
) -> Tensor: ...
def trunc_normal_(
    tensor: Tensor,
    mean: float = ...,
    std: float = ...,
    a: float = ...,
    b: float = ...,
    generator: torch.Generator | None = ...,
) -> Tensor: ...
def constant_(tensor: Tensor, val: float) -> Tensor: ...
def ones_(tensor: Tensor) -> Tensor: ...
def zeros_(tensor: Tensor) -> Tensor: ...
def eye_(tensor: Tensor) -> Tensor: ...
def dirac_(tensor: Tensor, groups: int = ...) -> Tensor: ...
def xavier_uniform_(
    tensor: Tensor, gain: float = ..., generator: torch.Generator | None = ...
) -> Tensor: ...
def xavier_normal_(
    tensor: Tensor, gain: float = ..., generator: torch.Generator | None = ...
) -> Tensor: ...
def kaiming_uniform_(
    tensor: Tensor,
    a: float = ...,
    mode: _FanMode = ...,
    nonlinearity: _NonlinearityType = ...,
    generator: torch.Generator | None = ...,
) -> Tensor: ...
def kaiming_normal_(
    tensor: Tensor,
    a: float = ...,
    mode: _FanMode = ...,
    nonlinearity: _NonlinearityType = ...,
    generator: torch.Generator | None = ...,
) -> Tensor: ...
def orthogonal_(
    tensor: Tensor, gain: float = ..., generator: torch.Generator | None = ...
) -> Tensor: ...
def sparse_(
    tensor: Tensor,
    sparsity: float,
    std: float = ...,
    generator: torch.Generator | None = ...,
) -> Tensor: ...

uniform = ...
normal = ...
constant = ...
eye = ...
dirac = ...
xavier_uniform = ...
xavier_normal = ...
kaiming_uniform = ...
kaiming_normal = ...
orthogonal = ...
sparse = ...

def _calculate_fan_in_and_fan_out(
    tensor: Tensor,
) -> tuple[int, int]: ...
