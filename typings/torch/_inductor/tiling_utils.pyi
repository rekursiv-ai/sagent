from collections.abc import Callable
from typing import Literal, TypeVar, overload

import dataclasses

from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode
from torch.utils._ordered_set import OrderedSet

import sympy

T = TypeVar("T")
U = TypeVar("U")
type Split = tuple[sympy.Expr, ...]
type VarsAndRanges = tuple[list[sympy.Symbol], list[sympy.Expr]]
loop_tiling_log = ...

def solve_for_zero(expr: sympy.Expr) -> sympy.Expr | None: ...
def solve_for_tiling(expr: sympy.Expr) -> sympy.Expr | None: ...
def find_coalesced_var(
    index: sympy.Expr, var_ranges: dict[sympy.Expr, int]
) -> sympy.Expr | None: ...

@dataclasses.dataclass(frozen=True)
class FusedNormalizedReadsWrites:
    index_vars: OrderedSet[sympy.Symbol]
    reduce_vars: OrderedSet[sympy.Symbol]
    reads: dict[sympy.Expr, OrderedSet[str]]
    writes: dict[sympy.Expr, OrderedSet[str]]
    var_ranges: dict[sympy.Symbol, int]

@overload
def get_pw_red_splits(
    n: SchedulerNode,
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: Literal[True],
) -> tuple[VarsAndRanges, VarsAndRanges] | None: ...
@overload
def get_pw_red_splits(
    n: SchedulerNode,
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: Literal[False] = ...,
) -> tuple[VarsAndRanges, VarsAndRanges]: ...
def get_pw_red_splits(
    n: SchedulerNode,
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: bool = ...,
) -> tuple[VarsAndRanges, VarsAndRanges] | None: ...

class NodeSplitGetter:
    def __init__(self, node: FusedSchedulerNode | SchedulerNode) -> None: ...
    def get_node_splits(self) -> tuple[Split, Split]: ...
    def try_split(self, pw: Split, red: Split) -> tuple[Split, Split] | None: ...

zip_equal = ...

def apply_var_mapping(
    iter_vars: list[sympy.Symbol],
    red_vars: list[sympy.Symbol],
    norm_pw_vars: list[sympy.Symbol],
    norm_red_vars: list[sympy.Symbol],
    new_ranges: list[list[sympy.Expr]],
    return_getters_groups: list[list[Callable[[list[sympy.Expr]], sympy.Expr]]],
) -> dict[sympy.Symbol, sympy.Expr]: ...
def extract_normalized_read_writes(
    node: FusedSchedulerNode | SchedulerNode,
) -> FusedNormalizedReadsWrites | None: ...
def get_score(addr: sympy.Expr, var_ranges: dict[sympy.Symbol, int]) -> int: ...
def get_hint(v: sympy.Expr | int) -> int: ...

@dataclasses.dataclass(frozen=True)
class VarTiling:
    var: sympy.Symbol
    tiling_factor: int
    score: int

@dataclasses.dataclass(frozen=True)
class CoalesceVarAnalysis:
    coalesced_by_var: dict[sympy.Expr, int]
    norm_read_writes: FusedNormalizedReadsWrites
    suggested_split: VarTiling | None = ...

def analyze_memory_coalescing(
    fused_node: FusedSchedulerNode | SchedulerNode,
) -> CoalesceVarAnalysis | None: ...
