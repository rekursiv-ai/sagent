from typing import NamedTuple

from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._sympy.value_ranges import ValueRanges

import sympy

__all__ = ["InputDim"]

class InputDim(NamedTuple):
    input_name: str
    dim: int

class _AddRuntimeAssertionsForInlineConstraintsPass(PassBase):
    def __init__(self, range_constraints: dict[sympy.Symbol, ValueRanges]) -> None: ...
    def call(self, graph_module) -> PassResult: ...
