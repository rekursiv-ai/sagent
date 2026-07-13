from typing import TypeVar

import torch

__all__ = [
    "fuse_conv_bn_eval",
    "fuse_conv_bn_weights",
    "fuse_linear_bn_eval",
    "fuse_linear_bn_weights",
]
ConvT = TypeVar("ConvT", bound=torch.nn.modules.conv._ConvNd)
LinearT = TypeVar("LinearT", bound=torch.nn.Linear)

def fuse_conv_bn_eval(
    conv: ConvT, bn: torch.nn.modules.batchnorm._BatchNorm, transpose: bool = ...
) -> ConvT: ...
def fuse_conv_bn_weights(
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: torch.Tensor | None,
    bn_b: torch.Tensor | None,
    transpose: bool = ...,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]: ...
def fuse_linear_bn_eval(
    linear: LinearT, bn: torch.nn.modules.batchnorm._BatchNorm
) -> LinearT: ...
def fuse_linear_bn_weights(
    linear_w: torch.Tensor,
    linear_b: torch.Tensor | None,
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]: ...
