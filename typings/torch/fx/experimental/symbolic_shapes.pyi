from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import _GeneratorContextManager, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Generic,
    NamedTuple,
    NoReturn,
    TypeGuard,
    TypeVar,
)
from typing_extensions import ParamSpec, deprecated

import abc
import functools
import inspect

from torch import SymBool, SymFloat, SymInt, Tensor
from torch._dynamo.source import TensorPropertySource
from torch._guards import ShapeGuard, SLoc, Source
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.recording import record_shapeenv_event
from torch.fx.experimental.sym_node import SymNode
from torch.types import BoolLikeType, FloatLikeType, IntLikeType
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.printers import CppPrinter, PythonPrinter
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._traceback import CapturedTraceback

import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree

InputList = list
DimList = list
log = ...

class GuardOnDataDependentSymNode(RuntimeError):
    cond: sympy.Basic
    def __init__(self, cond: sympy.Basic, *args: Any) -> None: ...

class PendingUnbackedSymbolNotFound(RuntimeError): ...

aten = ...
__all__ = [
    "CURRENT_NODE_KEY",
    "SHAPEENV_EVENT_KEY",
    "SYMPY_INTERP",
    "ConvertIntKey",
    "ShapeEnv",
    "Specialization",
    "StatefulSymbolicContext",
    "StatelessSymbolicContext",
    "SubclassSymbolicContext",
    "SymIntEqByExpr",
    "SymIntSymbolicContext",
    "SymbolicContext",
    "TrackedFake",
    "ValueRangesSLoc",
    "canonicalize_bool_expr",
    "check_consistent",
    "compute_unbacked_bindings",
    "create_contiguous",
    "free_symbols",
    "guard_float",
    "guard_int",
    "guard_or_false",
    "guard_or_true",
    "guard_scalar",
    "guard_size_oblivious",
    "has_free_symbols",
    "has_free_unbacked_symbols",
    "has_static_value",
    "has_symbolic_sizes_strides",
    "hint_int",
    "is_accessor_node",
    "is_concrete_bool",
    "is_concrete_float",
    "is_concrete_int",
    "is_nested_int",
    "is_symbol_binding_fx_node",
    "rebind_unbacked",
    "resolve_unbacked_bindings",
    "statically_known_false",
    "statically_known_true",
    "sym_and",
    "sym_eq",
    "sym_or",
]
SHAPEENV_EVENT_KEY = ...
CURRENT_NODE_KEY = ...

def log_lru_cache_stats(wrapped_f: functools._lru_cache_wrapper[object]) -> None: ...

type SympyBoolean = sympy.logic.boolalg.Boolean
_T = TypeVar("_T")
_SympyT = TypeVar("_SympyT", sympy.Expr, SympyBoolean, sympy.Basic)

class SymIntEqByExpr:
    def __init__(self, val: torch.SymInt | int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def lru_cache(
    maxsize: int | None,
) -> Callable[[Callable[..., _T]], functools._lru_cache_wrapper[_T]]: ...
@lru_cache(None)
def uninteresting_files() -> set[str]: ...

class ConstraintViolationError(RuntimeError): ...

def has_symbolic_sizes_strides(elem: torch.Tensor) -> bool: ...

type Int = torch.SymInt | int

def create_contiguous(shape: Sequence[Int]) -> list[Int]: ...
def hint_int(a: torch.SymInt | int, fallback: int | None = ...) -> int: ...

type Scalar = torch.SymInt | torch.SymFloat | torch.SymBool | int | float | bool

def has_hint(a: Scalar) -> bool: ...
def is_concrete_int(a: IntLikeType) -> bool: ...
def is_concrete_float(a: FloatLikeType) -> bool: ...
def is_concrete_bool(a: BoolLikeType) -> bool: ...
def has_static_value(a: SymBool | SymFloat | SymInt | bool | float) -> bool: ...
def guard_size_oblivious(expr: torch.SymBool | bool) -> bool: ...
def check_consistent(new: _T, old: _T) -> None: ...
def resolve_unbacked_bindings(
    shape_env: ShapeEnv | None, bindings: dict[sympy.Symbol, pytree.KeyPath] | None
) -> dict[sympy.Symbol, pytree.KeyPath] | None: ...

type Result = torch.Tensor | tuple[torch.Tensor, ...]

def rebind_unbacked(
    shape_env: ShapeEnv | None, n: torch.fx.Node, result: Result
) -> None: ...
def is_accessor_node(node: torch.fx.Node) -> bool: ...
def canonicalize_bool_expr(expr: _T) -> _T: ...
def is_nested_int(s: IntLikeType) -> TypeGuard[SymInt]: ...

type IterateExprsAtom = (
    SymInt | SymFloat | SymBool | int | float | bool | sympy.Basic | torch.Tensor
)
type IterateExprs = IterateExprsAtom | Sequence[IterateExprsAtom]

def free_symbols(val: IterateExprs) -> OrderedSet[sympy.Symbol]: ...
def has_free_symbols(val: IterateExprs) -> bool: ...
def has_free_unbacked_symbols(x: IterateExprs) -> bool: ...
def free_unbacked_symbols(x: IterateExprs) -> OrderedSet[sympy.Symbol]: ...
def is_symbol_binding_fx_node(node: torch.fx.Node) -> sympy.Symbol | None: ...
def find_symbol_binding_fx_nodes(
    graph: torch.fx.Graph,
) -> dict[sympy.Symbol, torch.fx.Node]: ...

@dataclass(frozen=True)
class Specialization:
    source: TensorPropertySource
    check_fn: Callable

@dataclass(frozen=True)
class ConvertIntKey:
    def get(self, b: bool) -> IntLikeType: ...

@dataclass(frozen=True)
class CallMethodKey:
    name: str

    def get(self, o: Any) -> Any: ...

@dataclass(frozen=True)
class InnerTensorKey:
    inner_name: str

    def get(self, o: Any) -> Any: ...

@dataclass(frozen=True)
class DivideByKey:
    divisor: IntLikeType

    def get(self, o: int) -> int: ...

def compute_unbacked_bindings(
    shape_env: ShapeEnv | None,
    example_value: object,
    old_example_value: object | None = ...,
    peek: bool = ...,
) -> dict[sympy.Symbol, pytree.KeyPath] | None: ...
def guard_or_false(a: BoolLikeType) -> bool: ...
def guard_or_true(a: BoolLikeType) -> bool: ...
def statically_known_false(x: BoolLikeType) -> bool: ...
def statically_known_true(x: BoolLikeType) -> bool: ...
def sym_and(x: BoolLikeType, *others: BoolLikeType) -> BoolLikeType: ...
def sym_eq(x: _T, y: _T) -> BoolLikeType: ...
def sym_or(x: BoolLikeType, *others: BoolLikeType) -> BoolLikeType: ...
def guard_scalar(
    a: SymBool | SymInt | SymFloat | bool | float,
) -> bool | int | float: ...
def constrain_range(a: SymInt, *, min: int | None, max: int | None = ...) -> None: ...
def constrain_unify(a: torch.SymInt, b: torch.SymInt) -> None: ...
def expect_true(a: BoolLikeType, skip: int = ...) -> bool: ...
def guard_bool(a: BoolLikeType) -> bool: ...
def guard_int(a: IntLikeType) -> int: ...
def guard_float(a: FloatLikeType) -> float: ...
def fx_placeholder_vals(gm: torch.fx.GraphModule) -> list[object]: ...
def fx_placeholder_targets(gm: torch.fx.GraphModule) -> list[str]: ...
def eval_guards(
    gm: torch.fx.GraphModule, *args: Tensor, ignore_static: bool = ...
) -> bool: ...
def bind_symbols(
    gm: torch.fx.GraphModule, *args: Tensor
) -> dict[sympy.Symbol, int]: ...

class DimDynamic(Enum):
    DYNAMIC = ...
    DUCK = ...
    STATIC = ...
    SIZE_LIKE_UNBACKED = ...
    INFER_STRIDE = ...
    OBLIVIOUS_SIZE = ...

@dataclass(frozen=True)
class Constraint:
    warn_only: bool

@dataclass(frozen=True)
class StrictMinMaxConstraint(Constraint):
    vr: ValueRanges
    def render(self, source: Source) -> str: ...

@dataclass(frozen=True)
class RelaxedUnspecConstraint(Constraint):
    def render(self, source: Source) -> str: ...

type DimConstraint = StrictMinMaxConstraint | RelaxedUnspecConstraint | None

@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    source_pairs: list[tuple[Source, Source]]
    derived_equalities: list[
        tuple[Source, Source | sympy.Symbol, Callable[[sympy.Expr], sympy.Expr]]
    ]
    phantom_symbols: list[sympy.Symbol]
    relaxed_sources: set[Source]
    _parents: dict[Source, Source] = ...
    _defs: dict[Source, sympy.Expr] = ...
    def __post_init__(self) -> None: ...
    def is_equal(self, source1: Source, source2: Source) -> bool: ...
    def is_derived(
        self, src: Source, symbol_src: Source, fn: Callable[[sympy.Expr], sympy.Expr]
    ) -> bool: ...

@dataclass(frozen=True)
class SymbolicContext: ...

@dataclass(frozen=True)
class SymIntSymbolicContext(SymbolicContext):
    constraint: DimConstraint

_P1 = ParamSpec("_P1")
_T1 = TypeVar("_T1")

@dataclass(frozen=True)
class StatelessSymbolicContext(SymbolicContext, Generic[_P1, _T1]):
    dynamic_sizes: DimList[DimDynamic]
    dynamic_strides: DimList[DimDynamic] = ...
    constraint_sizes: DimList[DimConstraint] = ...
    constraint_strides: DimList[DimConstraint] = ...
    specialize_on: list[list[Callable[_P1, _T1]]] | None = ...
    view_base_context: SymbolicContext | None = ...
    def __post_init__(self) -> None: ...

@dataclass(frozen=True)
class StatefulSymbolicContext(StatelessSymbolicContext):
    tensor_source: Source = ...
    shape_env_to_source_to_symbol_cache: dict[int, dict[str, sympy.Expr]] = ...
    def __post_init__(self) -> None: ...

@dataclass(frozen=True)
class SubclassSymbolicContext(StatefulSymbolicContext):
    inner_contexts: dict[str, SymbolicContext] = ...
    def __post_init__(self) -> None: ...

@dataclass
class TrackedFake:
    fake: FakeTensor | SymInt
    source: Source
    symbolic_context: SymbolicContext | None
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

def is_symbolic(
    val: SymInt | float | SymFloat | bool | SymBool,
) -> TypeGuard[SymInt | SymFloat | SymBool]: ...

IndicatorTypes = ...

@lru_cache(256)
def safe_expand(r: _SympyT) -> _SympyT: ...

class _SymbolInfo(NamedTuple):
    k: sympy.Symbol
    vr: ValueRanges | None
    val: sympy.Integer | None
    is_size_like: bool

def error() -> NoReturn: ...
def eval_is_non_overlapping_and_dense(
    sizes: Sequence[int], strides: Sequence[int]
) -> int: ...
def cast_symbool_to_symint_guardless(
    symbool: bool | torch.SymBool,
) -> int | torch.SymInt: ...

SYMPY_INTERP = ...

@dataclass(frozen=True)
class RuntimeAssert:
    expr: SympyBoolean
    msg: str = ...
    stack: CapturedTraceback = ...

class SymExprPrinter(PythonPrinter): ...

class _ShapeGuardPrinter(abc.ABC):
    def __init__(
        self,
        symbol_to_source: Mapping[sympy.Symbol, list[Source]],
        source_ref: Callable[[Source], str],
        var_to_sources: Mapping[sympy.Symbol, list[Source]],
    ) -> None: ...
    @abc.abstractmethod
    def print_source(self, source: Source) -> str: ...
    @abc.abstractmethod
    def doprint(self, expr: sympy.Expr) -> str: ...

class ShapeGuardPythonPrinter(_ShapeGuardPrinter, PythonPrinter):
    def __init__(self, *args: Any) -> None: ...
    def print_source(self, source: Source) -> str: ...
    def doprint(self, expr: sympy.Expr) -> str: ...

@deprecated(
    "`torch.fx.experimental.symbolic_shapes.ShapeGuardPrinter` is deprecated, please use `torch.fx.experimental.symbolic_shapes.ShapeGuardPythonPrinter` instead.",
    category=FutureWarning,
)
class ShapeGuardPrinter(ShapeGuardPythonPrinter): ...

class _ShapeGuardCppPrinter(_ShapeGuardPrinter, CppPrinter):
    def __init__(self, *args: Any) -> None: ...
    def print_source(self, source: Source) -> str: ...
    def doprint(self, expr: sympy.Expr) -> str: ...

@dataclass(frozen=True)
class _ShapeGuardsHelper:
    exprs: list[str]

@dataclass(frozen=True)
class _CppShapeGuardsHelper(_ShapeGuardsHelper):
    source_to_symbol: dict[Source, sympy.Symbol]

class LoggingShapeGuardPrinter(ShapeGuardPythonPrinter):
    def __init__(self, var_to_sources: Mapping[sympy.Symbol, list[Source]]) -> None: ...

class DynamicDimConstraintPrinter(PythonPrinter):
    def __init__(
        self,
        symbol_to_source: dict[sympy.Symbol, list[Source]],
        source_name_to_debug_name: Mapping[str, str],
    ) -> None: ...

class DimConstraints:
    def __init__(
        self,
        symbol_to_source: dict[sympy.Symbol, list[Source]],
        var_to_val: Mapping[sympy.Symbol, sympy.Integer],
        marked_dynamic: set[sympy.Symbol],
        source_name_to_debug_name: Mapping[str, str],
    ) -> None: ...
    def rewrite_with_congruences(self, s: sympy.Symbol, expr: _SympyT) -> _SympyT: ...
    def add(self, expr: SympyBoolean) -> bool: ...
    def add_equality(self, source: Source, expr: sympy.Expr) -> None: ...
    def solve(self) -> None: ...
    def forced_specializations(self) -> dict[str, sympy.Expr]: ...
    def prettify_results(
        self,
        original_signature: inspect.Signature,
        dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any],
        constraint_violation_error: object,
        forced_specializations: dict[str, str],
    ) -> str: ...

TLS = ...

@dataclass(frozen=True)
class ShapeEnvSettings:
    allow_scalar_outputs: bool
    allow_dynamic_output_shape_ops: bool
    assume_static_by_default: bool
    specialize_zero_one: bool
    duck_shape: bool
    prefer_deferred_runtime_asserts_over_guards: bool
    trace_asserts: bool

@dataclass
class ValueRangesSLoc:
    lower: SLoc
    upper: SLoc

@dataclass
class _FrameLocalResult:
    loc: str | None = ...
    locals: dict[str, Any] = ...
    symbols: dict[str, str] = ...

class ShapeEnv:
    def __init__(
        self,
        *,
        should_record_events: bool | None = ...,
        tracked_fakes: list[Any] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    @property
    def allow_scalar_outputs(self) -> bool: ...
    @property
    def allow_dynamic_output_shape_ops(self) -> bool: ...
    @property
    def assume_static_by_default(self) -> bool: ...
    @property
    def specialize_zero_one(self) -> bool: ...
    @property
    def duck_shape(self) -> bool: ...
    @property
    def prefer_deferred_runtime_asserts_over_guards(self) -> bool: ...
    @contextmanager
    def patch_source_specialization(
        self, source: Source, check_fn: Callable[[sympy.Symbol], sympy.Expr]
    ) -> Iterator[None]: ...
    def check_equal(self, other: ShapeEnv) -> None: ...
    @record_shapeenv_event()
    def set_unbacked_var_to_val(self, k: sympy.Symbol, v: int) -> None: ...
    @contextmanager
    def ignore_fresh_unbacked_symbols(self) -> Iterator[None]: ...
    @record_shapeenv_event()
    def freeze(self) -> None: ...
    @record_shapeenv_event()
    def freeze_runtime_asserts(self) -> None: ...
    def suppress_guards(self) -> _GeneratorContextManager[None]: ...
    def create_symbolic_sizes_strides_storage_offset(
        self,
        ex: torch.Tensor,
        source: Source,
        *,
        symbolic_context: SymbolicContext | None = ...,
    ) -> tuple[tuple[IntLikeType, ...], tuple[IntLikeType, ...], IntLikeType]: ...
    @record_shapeenv_event()
    def create_symintnode(
        self, sym: sympy.Expr, *, hint: int | None, source: Source | None = ...
    ) -> IntLikeType: ...
    @record_shapeenv_event()
    def create_symfloatnode(
        self, sym: sympy.Expr, *, hint: float | bool | None, source: Source | None = ...
    ) -> FloatLikeType: ...
    @record_shapeenv_event()
    def create_unspecified_symint_and_symbol(
        self, value: int, source: Source, dynamic_dim: DimDynamic
    ) -> IntLikeType: ...
    def create_symboolnode(self, sym: sympy.Expr) -> SymBool: ...
    @record_shapeenv_event()
    def create_unbacked_symfloat(self) -> SymFloat: ...
    @record_shapeenv_event()
    def create_unbacked_symint(self, source: Source | None = ...) -> SymInt: ...
    def is_unbacked_symint(self, symbol: sympy.Symbol) -> bool: ...
    @record_shapeenv_event()
    def create_unbacked_symbool(self) -> SymBool: ...
    @record_shapeenv_event()
    def create_unspecified_symbol(
        self,
        val: SymInt | float | SymFloat,
        source: Source,
        dynamic_dim: DimDynamic = ...,
        constraint_dim: DimConstraint = ...,
        symbolic_context: StatelessSymbolicContext | None = ...,
    ) -> sympy.Expr: ...
    @record_shapeenv_event()
    def create_symbol(
        self,
        val: int,
        source: Source,
        dynamic_dim: DimDynamic = ...,
        constraint_dim: DimConstraint = ...,
        positive: bool | None = ...,
        do_not_specialize_zero_one: bool = ...,
        symbolic_context: StatelessSymbolicContext | None = ...,
    ) -> sympy.Expr: ...
    def add_var_to_val(self, expr: sympy.Symbol, val: int) -> None: ...
    def produce_guards(self, *args: Any, **kwargs: Any) -> list[str]: ...
    def produce_guards_verbose(
        self,
        placeholders: Sequence[FakeTensor],
        sources: Sequence[Source],
        source_ref: Callable[[Source], str] = ...,
        *,
        guards: list[ShapeGuard] | None = ...,
        input_contexts: DimList[SymbolicContext] | None = ...,
        equalities_inputs: EqualityConstraint | None = ...,
        _simplified: bool = ...,
        ignore_static: bool = ...,
        langs: tuple[str, ...] = ...,
    ) -> list[_ShapeGuardsHelper]: ...
    def produce_guards_expression(
        self,
        placeholders: Sequence[SymInt | FakeTensor],
        *,
        guards: list[ShapeGuard] | None = ...,
        ignore_static: bool = ...,
    ) -> str | None: ...
    def evaluate_symexpr(self, code: str) -> int | float | bool: ...
    def deserialize_symexpr(self, code: str) -> SymInt | SymFloat | SymBool: ...
    def evaluate_guards_expression(self, code: str, args: Sequence[object]) -> bool: ...
    def evaluate_guards_for_args(
        self,
        placeholders: Sequence[FakeTensor],
        args: Sequence[Tensor],
        *,
        ignore_static: bool = ...,
    ) -> bool: ...
    def get_pruned_guards(
        self, symints: Sequence[torch.SymInt]
    ) -> list[ShapeGuard]: ...
    def bind_symbols(
        self, placeholders: Sequence[FakeTensor], args: Sequence[Tensor]
    ) -> dict[sympy.Symbol, int]: ...
    def get_nontrivial_guards(self) -> list[SympyBoolean]: ...
    def format_guards(self, verbose: bool = ...) -> str: ...
    def bound_sympy(
        self, expr: sympy.Expr, size_oblivious: bool = ...
    ) -> ValueRanges: ...
    @_lru_cache
    def get_axioms(
        self, symbols: tuple[sympy.Symbol] | None = ..., compute_hint: bool = ...
    ) -> tuple[SympyBoolean, ...]: ...
    @lru_cache(None)
    def get_implications(
        self, e: SympyBoolean
    ) -> tuple[tuple[SympyBoolean, sympy.logic.boolalg.BooleanAtom], ...]: ...
    @_lru_cache
    def replace(self, expr: _SympyT) -> _SympyT: ...
    @_lru_cache
    def simplify(self, expr: _SympyT, size_oblivious: bool = ...) -> _SympyT: ...
    @lru_cache(256)
    def size_hint(
        self, expr: sympy.Basic, *, allow_none: bool = ...
    ) -> sympy.Basic | None: ...
    @lru_cache(256)
    def has_hint(self, expr: sympy.Expr) -> bool: ...

    _expr_sym_node_id: int | None = ...
    def evaluate_sym_node(
        self,
        sym_node: SymNode,
        size_oblivious: bool = ...,
        fallback_value: bool | None = ...,
    ) -> sympy.Basic: ...
    def evaluate_expr(
        self,
        orig_expr: sympy.Basic,
        hint: bool | float | None = ...,
        fx_node: torch.fx.Node | None = ...,
        size_oblivious: bool = ...,
        fallback_value: bool | None = ...,
        *,
        forcing_spec: bool = ...,
    ) -> sympy.Basic: ...
    def cleanup(self) -> None: ...
    @lru_cache(256)
    @record_shapeenv_event(save_tracked_fakes=True)
    def guard_or_defer_runtime_assert(
        self, orig_expr: SympyBoolean, msg: str, fx_node: torch.fx.Node | None = ...
    ) -> bool: ...
    @lru_cache(maxsize=None)
    @record_shapeenv_event()
    def constrain_symbol_range(
        self, s: sympy.Symbol, compiler_min: int, compiler_max: int
    ) -> None: ...

class PropagateUnbackedSymInts(torch.fx.Interpreter):
    def run_node(self, n: torch.fx.Node) -> Result: ...

class _PythonMsgPrinter(PythonPrinter):
    def __init__(self, src_map: dict[str, list[str]]) -> None: ...
