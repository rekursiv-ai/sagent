from typing import Self

from torch import Tensor
from torch.distributions.gamma import Gamma

__all__ = ["Chi2"]

class Chi2(Gamma):
    def __init__(
        self, df: Tensor | float, validate_args: bool | None = ...
    ) -> None: ...
    def expand(self, batch_shape, _instance=...) -> Self: ...
    @property
    def df(self) -> Tensor: ...
