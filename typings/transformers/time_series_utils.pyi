from collections.abc import Callable

from torch import nn
from torch.distributions import Distribution, TransformedDistribution

import torch

"""
Time series distributional output classes and utilities.
"""

class AffineTransformed(TransformedDistribution):
    def __init__(
        self, base_distribution: Distribution, loc=..., scale=..., event_dim=...
    ) -> None: ...
    @property
    def mean(self):  # -> Tensor:
        ...
    @property
    def variance(self):  # -> Tensor:
        ...
    @property
    def stddev(self):  # -> Tensor:
        ...

class ParameterProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        args_dim: dict[str, int],
        domain_map: Callable[..., tuple[torch.Tensor]],
        **kwargs,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]: ...

class LambdaLayer(nn.Module):
    def __init__(self, function) -> None: ...
    def forward(self, x, *args): ...

class DistributionOutput:
    distribution_class: type
    in_features: int
    args_dim: dict[str, int]
    def __init__(self, dim: int = ...) -> None: ...
    def distribution(
        self,
        distr_args,
        loc: torch.Tensor | None = ...,
        scale: torch.Tensor | None = ...,
    ) -> Distribution: ...
    @property
    def event_shape(self) -> tuple: ...
    @property
    def event_dim(self) -> int: ...
    @property
    def value_in_support(self) -> float: ...
    def get_parameter_projection(self, in_features: int) -> nn.Module: ...
    def domain_map(self, *args: torch.Tensor): ...
    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor: ...

class StudentTOutput(DistributionOutput):
    args_dim: dict[str, int] = ...
    distribution_class: type = ...
    @classmethod
    def domain_map(
        cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ):  # -> tuple[Tensor, Tensor, Tensor]:
        ...

class NormalOutput(DistributionOutput):
    args_dim: dict[str, int] = ...
    distribution_class: type = ...
    @classmethod
    def domain_map(
        cls, loc: torch.Tensor, scale: torch.Tensor
    ):  # -> tuple[Tensor, Tensor]:
        ...

class NegativeBinomialOutput(DistributionOutput):
    args_dim: dict[str, int] = ...
    distribution_class: type = ...
    @classmethod
    def domain_map(
        cls, total_count: torch.Tensor, logits: torch.Tensor
    ):  # -> tuple[Tensor, Tensor]:
        ...
    def distribution(
        self,
        distr_args,
        loc: torch.Tensor | None = ...,
        scale: torch.Tensor | None = ...,
    ) -> Distribution: ...
