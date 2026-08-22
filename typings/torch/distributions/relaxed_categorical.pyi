from typing import Self

from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.types import _size

import torch

__all__ = ["ExpRelaxedCategorical", "RelaxedOneHotCategorical"]

class ExpRelaxedCategorical(Distribution):
    has_rsample = ...
    def __init__(
        self,
        temperature: Tensor,
        probs: Tensor | None = ...,
        logits: Tensor | None = ...,
        validate_args: bool | None = ...,
    ) -> None: ...
    def expand(self, batch_shape, _instance=...) -> Self: ...
    @property
    def param_shape(self) -> torch.Size: ...
    @property
    def logits(self) -> Tensor: ...
    @property
    def probs(self) -> Tensor: ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value) -> Tensor: ...

class RelaxedOneHotCategorical(TransformedDistribution):
    base_dist: ExpRelaxedCategorical
    def __init__(
        self,
        temperature: Tensor,
        probs: Tensor | None = ...,
        logits: Tensor | None = ...,
        validate_args: bool | None = ...,
    ) -> None: ...
    def expand(self, batch_shape, _instance=...) -> Self: ...
    @property
    def temperature(self) -> Tensor: ...
    @property
    def logits(self) -> Tensor: ...
    @property
    def probs(self) -> Tensor: ...
