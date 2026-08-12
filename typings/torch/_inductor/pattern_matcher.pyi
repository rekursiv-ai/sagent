from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from typing import Any, Protocol, TypeVar
from typing_extensions import TypeIs

import dataclasses
import functools

from torch.utils._ordered_set import OrderedSet

import torch
import torch.fx
import torch.utils._pytree as pytree

from .._functorch import config as functorch_config

"""
# Inductor Pattern Matcher

The pattern matcher enables search/replace within an FX graph.

The main entrypoint to the pattern matcher is register_replacement(). Given a
search function and a replacement function this will register a replacement with
a pass (such as torch._inductor.fx_passes.joint_graph.patterns).

Internally the pattern matcher represents patterns as a graph (a DAG). Creating
new patterns manually as a graph is cumbersome and error-prone so the standard
way to create patterns (using register_replacement()) is to provide a search
function and a replacement function which is traced and converted into a graph.

Because the search functions are built somewhat generic (they tend to ignore
tensor sizes, for example) register_replacement() allows you to specify an
`extra_check` function which performs additional checks to verify that the
matched pattern fully matches before returning it.

## Precompiled Patterns

New patterns are added using register_replacement(). Patterns added in this way
can have a compile-time overhead because they need to be traced before
use. Patterns can be precompiled and added using gen_register_replacement()
instead. To do this you call gen_register_replacement() instead of
register_replacement(). The arguments are the same except for an additional
unique name which is used as a lookup key.

## Internals

The match DAG is represented by a graph of `PatternExpr` nodes. Each PatternExpr
implements a `_match` method which returns either a `Match` object for a
successful match or a `FailedMatch` object for a failure to match.
"""
log = ...
aten = ...
prims = ...
type Constant = Any
type NodeOrConstant = Constant | torch.fx.Node
backend = ...

class SearchFn(Protocol):
    __name__: str
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class ReplaceFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class TraceFn(Protocol):
    def __call__(
        self, fn: SearchFn | ReplaceFn, *args: Any, **kwargs: Any
    ) -> torch.fx.GraphModule: ...

T = TypeVar("T")
type FnsType = torch.fx.node.Target | str

class Multiple:
    def __init__(self) -> None: ...

MULTIPLE = ...

class Match:
    pattern: PatternExpr
    args: list[Any]
    kwargs: dict[str, Any]
    nodes: list[torch.fx.Node]
    targets: dict[_TargetExpr, torch.fx.node.Target]
    ctx: MatchContext
    replacement_graph: torch.fx.GraphModule | None
    def __init__(
        self,
        ctx: MatchContext,
        pattern: PatternExpr,
        args: Sequence[Any] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ) -> None: ...
    @property
    def graph(self) -> torch.fx.Graph: ...
    def extend(self, other: Match) -> None: ...
    def bundle(self) -> Match: ...
    def erase_nodes(self) -> None: ...
    def output_nodes(self) -> list[torch.fx.Node | None]: ...
    def output_node(self) -> torch.fx.Node: ...
    def replace_with_graph(
        self, replacement_graph: torch.fx.Graph, args: Sequence[Any]
    ) -> None: ...
    def replace_by_example(
        self,
        replacement_fn: ReplaceFn,
        args: Sequence[Any],
        trace_fn: TraceFn | None = ...,
        run_functional_passes: bool = ...,
    ) -> None: ...

class FailedMatch(RuntimeError):
    format_string: str
    def __init__(self, format_string: str, *args: Any, **kwargs: Any) -> None: ...
    def __bool__(self) -> bool: ...

type MatchResult = Match | FailedMatch

def is_match(m: MatchResult) -> TypeIs[Match]: ...

class MatchContext:
    outputs: list[PatternExpr | None]
    pattern_to_node: dict[PatternExpr, torch.fx.Node | None]
    graph: torch.fx.Graph
    exclusive_node_set: list[NodeOrConstant]
    def __init__(
        self,
        outputs: list[PatternExpr | None],
        pattern_to_node: dict[PatternExpr, torch.fx.Node] | None = ...,
        *,
        graph: torch.fx.Graph,
    ) -> None: ...
    def match(self, pattern: PatternExpr, node: NodeOrConstant) -> MatchResult: ...
    def filter_multi_user_patterns(self) -> dict[PatternExpr, torch.fx.Node]: ...

class PatternExpr(ABC):
    def match(self, node: torch.fx.Node) -> MatchResult: ...
    def has_multiple_users(self) -> bool: ...
    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[torch.fx.Node | None]: ...
    def pattern_eq(self, other: Any) -> bool: ...

class Arg(PatternExpr): ...

class Ignored(PatternExpr):
    def pretty_print(self, pp: PatternPrettyPrinter) -> str: ...

class KeywordArg(PatternExpr):
    def __init__(self, name: str) -> None: ...
    def pattern_eq(self, other: Any) -> bool: ...

class ExclusiveKeywordArg(PatternExpr):
    name: str
    def __init__(self, name: str) -> None: ...
    def pattern_eq(self, other: Any) -> bool: ...

class _TargetExpr(PatternExpr):
    fns: list[FnsType]
    fns_set: OrderedSet[FnsType]
    def __init__(
        self, fns: FnsType | Sequence[FnsType], users: Multiple | int = ...
    ) -> None: ...
    @property
    @abstractmethod
    def op(self) -> str: ...
    def fns_repr(self) -> str: ...
    def has_multiple_users(self) -> bool: ...
    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[torch.fx.Node | None]: ...
    def pattern_eq(self, other: Any) -> bool: ...

type _SimpleSpec = tuple[Any, ...]

class _TargetArgsExpr(_TargetExpr):
    def __init__(
        self,
        fns: torch.fx.node.Target | str | Sequence[Any],
        *args: Any,
        _users: int | Multiple = ...,
        **kwargs: Any,
    ) -> None: ...
    @staticmethod
    def simple_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> tuple[Sequence[Any], _SimpleSpec | pytree.TreeSpec]: ...
    @staticmethod
    def pytree_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> tuple[Sequence[Any], _SimpleSpec | pytree.TreeSpec]: ...
    def pretty_print(self, pp: PatternPrettyPrinter) -> str: ...
    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[torch.fx.Node | None]: ...
    def pattern_eq(self, other: Any) -> bool: ...

class CallFunction(_TargetArgsExpr): ...
class CallMethod(_TargetArgsExpr): ...
class CallModule(_TargetArgsExpr): ...
class _TargetExprVarArgs(_TargetExpr): ...
class CallFunctionVarArgs(_TargetExprVarArgs): ...
class CallMethodVarArgs(_TargetExprVarArgs): ...
class CallModuleVarArgs(_TargetExprVarArgs): ...

class ListOf(PatternExpr):
    def __init__(self, pattern: PatternExpr, partial: bool = ...) -> None: ...
    def pattern_eq(self, other: Any) -> bool: ...

class MultiOutputPattern(PatternExpr):
    outputs: list[PatternExpr | None]
    def __init__(self, outputs: Sequence[PatternExpr | None]) -> None: ...
    @property
    def fns(self) -> Callable[..., Any] | str | Sequence[Any]: ...
    def pretty_print(self, pp: PatternPrettyPrinter) -> str: ...
    def match(self, node: torch.fx.Node) -> MatchResult: ...
    def pattern_eq(self, other: Any) -> bool: ...

class RepeatedExpr(PatternExpr):
    def __init__(self, inner_pattern: _TargetExpr) -> None: ...
    @property
    def fns(self) -> Sequence[FnsType]: ...
    def pattern_eq(self, other: Any) -> bool: ...

class PatternPrettyPrinter:
    def __init__(self) -> None: ...
    @staticmethod
    @functools.cache
    def run(obj: PatternExpr, output_name: str = ...) -> str: ...
    def pretty_print(self, obj: Any) -> str: ...
    def memoize(self, obj: _TargetArgsExpr) -> str: ...

class _PassDictsType(Protocol):
    def __getitem__(
        self, k: tuple[str, torch.fx.node.Target]
    ) -> list[PatternEntry]: ...

@dataclasses.dataclass
class PatternEntry:
    pattern: PatternExpr
    extra_check: Callable[[Match], bool]
    def apply(
        self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node
    ) -> None: ...
    def register(
        self,
        pass_dicts: _PassDictsType | Sequence[_PassDictsType],
        target: torch.fx.node.Target | None = ...,
        prepend: bool = ...,
    ) -> None: ...

@dataclasses.dataclass
class LoweringPatternEntry(PatternEntry):
    handler: Callable[..., Any]
    def apply(
        self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node
    ) -> None: ...

@dataclasses.dataclass
class GraphPatternEntry(PatternEntry):
    handler: Callable[..., Any]
    def apply(
        self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node
    ) -> None: ...

@dataclasses.dataclass
class ReplacementPatternEntry(PatternEntry):
    normalize_args: Callable[..., list[Any]]
    @staticmethod
    def replace_with_graph(
        match: Match,
        graph: torch.fx.Graph,
        replacement_graph: torch.fx.Graph | torch.fx.GraphModule,
        args: Sequence[torch.fx.Node],
    ) -> None: ...
    def apply(
        self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node
    ) -> None: ...

def log_trace_failure(search_fn: Callable[..., Any], e: RuntimeError) -> None: ...
def check_and_add_duplicate_pattern(
    pattern: PatternExpr,
    graph: torch.fx.Graph | None,
    seen_patterns: dict[str, list[str | None]],
    skip_duplicates: bool = ...,
) -> bool: ...
def register_replacement(
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: _PassDictsType | Sequence[_PassDictsType],
    extra_check: Callable[[Match], bool] = ...,
    scalar_workaround: dict[str, float | int] | None = ...,
    exclusive_arg_names: Sequence[str] = ...,
    search_fn_pattern: PatternExpr | None = ...,
    skip_duplicates: bool = ...,
) -> bool: ...

_serialized_patterns: OrderedSet[str] = ...
SERIALIZED_PATTERN_PATH = ...
_known_precompiled_patterns: list[
    tuple[
        Any,
        Iterable[Any],
        Callable[[Callable[..., Any], Iterable[Any]], torch.fx.GraphModule],
        Any,
        PatternExpr,
    ]
] = ...

def gen_register_replacement(
    unique_name: str,
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: _PassDictsType | Sequence[_PassDictsType],
    extra_check: Callable[[Match], bool] = ...,
    scalar_workaround: dict[str, float | int] | None = ...,
    exclusive_arg_names: Sequence[str] = ...,
    skip_duplicates: bool = ...,
) -> None: ...
@functorch_config.patch(functionalize_rng_ops=False)
def gen_pattern_and_search_gm(
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: dict[str, float | int] | None = ...,
    exclusive_arg_names: Sequence[str] = ...,
) -> tuple[PatternExpr, torch.fx.GraphModule]: ...
def gen_pattern(
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: dict[str, float | int] | None = ...,
    exclusive_arg_names: Sequence[str] = ...,
) -> PatternExpr: ...
def register_lowering_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = ...,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = ...,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
def register_graph_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = ...,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = ...,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
def is_start_of_fx_graph(graph: torch.fx.Graph, node: torch.fx.Node) -> bool: ...

_mutation_op_re = ...

def fixme_incorrect_inductor_schema_op(op: torch._ops.OpOverload) -> bool: ...
def is_mutation_op(node: torch.fx.Node) -> bool: ...
def same_mutation_regions(a: torch.fx.Node, b: torch.fx.Node) -> bool: ...
def get_mutation_region_id(graph: torch.fx.Graph, node: torch.fx.Node) -> int: ...
def should_compute_mutation_region_ids(graph: torch.fx.Graph) -> bool: ...
def compute_mutation_region_ids(graph: torch.fx.Graph) -> None: ...

class PatternMatcherPass:
    def __init__(self, pass_name: str | None = ...) -> None: ...
    def __getitem__(
        self, item: tuple[str, torch.fx.node.Target]
    ) -> list[PatternEntry]: ...
    def apply(self, gm: torch.fx.GraphModule | torch.fx.Graph) -> int: ...
    def clear(self) -> None: ...

def fx_to_pattern(
    gm: torch.fx.GraphModule | torch.fx.Graph,
    ignore_types: Sequence[type[Any]] = ...,
    argnames: Sequence[str] = ...,
    scalar_workaround: dict[str, float | int] | None = ...,
    exclusive_arg_names: Sequence[str] = ...,
) -> PatternExpr: ...
@torch.no_grad()
def fwd_only(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    run_functional_passes: bool = ...,
    get_decomp_fn: Callable[..., Any] | None = ...,
) -> torch.fx.GraphModule: ...
@torch.enable_grad()
def joint_fwd_bwd(
    fn: Callable[..., Any], args: Sequence[Any]
) -> torch.fx.GraphModule: ...
def stable_topological_sort(graph: torch.fx.Graph) -> None: ...
def init_once_fakemode(fn: Callable[..., Any]) -> Callable[[], Any]: ...
def config_flag(name: str) -> Callable[[Match], Any]: ...
def clone_graph(input_graph: torch.fx.GraphModule) -> torch.fx.GraphModule: ...

_seen_patterns: OrderedSet[str] = ...

def get_arg_value(
    node: torch.fx.Node, arg_number: int, kwarg_name: str | None = ...
) -> Any: ...
def filter_nodes(nodes: Iterable[torch.fx.Node], fn: Any) -> list[torch.fx.Node]: ...
def extract_target(node: torch.fx.Node) -> torch.fx.node.Target: ...
