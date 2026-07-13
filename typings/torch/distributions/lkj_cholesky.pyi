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
    arg_constraints = ...
    support = ...
    def __init__(
        self,
        dim: int,
        concentration: Tensor | float = ...,
        validate_args: bool | None = ...,
    ) -> None: ...
    def expand(self, batch_shape, _instance=...) -> Self: ...
    def sample(self, sample_shape=...) -> Any | Tensor: ...
    def log_prob(self, value) -> Tensor: ...
