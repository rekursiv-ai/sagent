from typing import Self

from torch import Tensor
from torch.distributions.distribution import Distribution

"""
This closely follows the implementation in NumPyro (https://github.com/pyro-ppl/numpyro).

Original copyright notice:

# Copyright: Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
"""
__all__ = ["LKJCholesky"]

class LKJCholesky(Distribution):
    def __init__(
        self,
        dim: int,
        concentration: Tensor | float = ...,
        validate_args: bool | None = ...,
    ) -> None: ...
    def expand(self, batch_shape, _instance=...) -> Self: ...
    def sample(self, sample_shape=...) -> Tensor: ...
    def log_prob(self, value) -> Tensor: ...
