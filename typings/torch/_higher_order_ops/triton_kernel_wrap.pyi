from collections.abc import Callable, Sequence
from typing import Any, Never

import dataclasses

from torch import SymInt
from torch._C import DispatchKey
from torch._dynamo.symbolic_convert import InstructionTranslator
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.functions import TritonKernelVariable
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import BaseFunctionalizeAPI
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.proxy import Proxy
from torch.types import IntLikeType
from triton._C.libtriton.ir import module as TritonIRModule

import sympy

type TritonMetaParamsType = dict[str, int]
type TritonGridTupleType = tuple[int | sympy.Expr | SymInt, ...]
type TritonGridCallableType = Callable[[TritonMetaParamsType], tuple[int, ...]]
type TritonGridType = TritonGridTupleType | TritonGridCallableType
type TritonKernelType = Autotuner | JITFunction
type TritonAutotunerType = Autotuner
log = ...
type TMAExperimentalMetadata = tuple[
    str, tuple[list[IntLikeType], list[IntLikeType], IntLikeType]
]
type TMAStableMetadata = tuple[str, tuple[list[IntLikeType],]]

def create_tma_experimental_metadata(
    dims: list[IntLikeType], block_dims: list[IntLikeType], element_size: IntLikeType
) -> TMAExperimentalMetadata: ...
def maybe_unpack_tma_experimental_metadata(
    tma_meta: TMAExperimentalMetadata | TMAStableMetadata,
) -> tuple[list[IntLikeType], list[IntLikeType], IntLikeType] | None: ...
def create_tma_stable_metadata(block_shape: list[IntLikeType]) -> TMAStableMetadata: ...
def maybe_unpack_tma_stable_metadata(
    tma_meta: TMAExperimentalMetadata | TMAStableMetadata,
) -> tuple[list[IntLikeType]] | None: ...

type TMADescriptorMetadata = dict[str, TMAExperimentalMetadata | TMAStableMetadata]

class KernelSideTable:
    id_to_kernel: dict[int, TritonKernelType] = ...
    kernel_to_id: dict[TritonKernelType, int] = ...
    constant_args: dict[int, dict[str, Any]] = ...
    lock = ...
    def add_kernel(self, kernel: TritonKernelType) -> int: ...
    def get_kernel(self, idx: int) -> TritonKernelType: ...
    def add_constant_args(self, args: dict[str, Any]) -> int: ...
    def get_constant_args(self, idx: int) -> dict[str, Any]: ...
    def reset_table(self) -> None: ...

kernel_side_table = ...

@dataclasses.dataclass(frozen=True)
class Param:
    idx: int

@dataclasses.dataclass(frozen=True)
class Intermediate:
    idx: int
    def fake(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class Op:
    name: str
    fn_call_name: str | None
    args: list[Param | Intermediate]
    ret: Intermediate = ...
    sub_idx: int | None = ...
    is_pure: bool = ...
    def __post_init__(self) -> None: ...

def generate_ttir(
    kernel: TritonKernelType,
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> tuple[TritonIRModule, list[str]]: ...
def ttir_to_functions(
    ttir_module: TritonIRModule,
) -> dict[str, dict[Intermediate, list[Op]]]: ...

class MemoizeWithCycleCheck:
    fn: Callable[..., Any]
    cache: dict[tuple[Any], Any]
    def __init__(self, fn: Callable[..., Any]) -> None: ...
    def __call__(
        self,
        functions: dict[str, dict[Intermediate, list[Op]]],
        fn_name: str,
        *args: Any,
    ) -> list[bool]: ...
    def reset(self) -> None: ...

@MemoizeWithCycleCheck
def get_tma_stores(
    functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str
) -> set[Intermediate | Param]: ...
@MemoizeWithCycleCheck
def analyze_kernel_mutations(
    functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str, num_args: int
) -> list[bool]: ...
def identify_mutated_tensors(
    kernel: TritonKernelType,
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> list[str]: ...

class TritonKernelWrapperMutation(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        grid: list[TritonGridType],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: dict[str, Any],
    ) -> Any: ...

triton_kernel_wrapper_mutation = ...

class TritonKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        grid: list[TritonGridType],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: dict[str, Any],
        tensors_to_clone: list[str],
    ) -> dict[str, Any]: ...

triton_kernel_wrapper_functional = ...

@triton_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_mutation_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list[TritonGridType],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None: ...
@triton_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def triton_kernel_wrapper_mutation_fake_tensor_mode(
    mode: FakeTensorMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list[TritonGridType],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None: ...
@triton_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
def _(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list[TritonGridType],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None: ...
def trace_triton_kernel_wrapper(
    proxy_mode: ProxyTorchDispatchMode,
    func_overload: Callable[..., Any],
    node_args: dict[str, Any],
) -> dict[str, Any] | None: ...
@triton_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list[TritonGridType],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None: ...
def get_mutated_tensors(
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> list[str]: ...
@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(
    ctx: BaseFunctionalizeAPI,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list[TritonGridType],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None: ...
@triton_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_functional_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list[TritonGridType],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
) -> dict[str, Any]: ...
@triton_kernel_wrapper_functional.py_impl(FakeTensorMode)
def triton_kernel_wrapper_functional_fake_tensor_mode(
    mode: FakeTensorMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list[TritonGridType],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
) -> dict[str, Any]: ...
@triton_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_functional_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list[TritonGridType],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
) -> dict[str, Any]: ...
@triton_kernel_wrapper_functional.py_functionalize_impl
def triton_kernel_wrapper_functional_functionalize(
    ctx: BaseFunctionalizeAPI,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list[TritonGridType],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
) -> dict[str, Any]: ...

class TritonHOPifier:
    def raise_unsupported(self, msg: str) -> Never: ...
    def is_callable(self, maybe_callable: Any) -> bool: ...
    def get_value(self, val: Any) -> Any: ...
    def call_grid(
        self, grid, meta, tx
    ) -> tuple[int | sympy.Expr | SymInt, ...] | tuple[Proxy, ...]: ...
    def wrap_user_defined_obj(
        self,
        user_obj: Any,
        tx: InstructionTranslator | None,
        variable: TritonKernelVariable | TraceableTritonKernelWrapper | None,
        name: str,
    ) -> Any: ...
    def call_user_defined_fn(
        self,
        user_fn: Callable[..., Any],
        args: list,
        kwargs: dict,
        tx: InstructionTranslator | None,
        variable: TritonKernelVariable | TraceableTritonKernelWrapper | None,
    ) -> Any: ...
    def maybe_unpack_configs(
        self, configs: list[TritonConfig], tx: InstructionTranslator | None
    ) -> list[TritonConfig]: ...
    def maybe_unpack_heuristic_result(self, result: Any) -> Any: ...
    @staticmethod
    def do_prune_configs(
        autotuner: TritonAutotunerType,
        early_config_prune: Callable | None,
        perf_model: Callable | None,
        top_k: float,
        configs: list,
        named_args: dict,
        kwargs: dict,
    ) -> list[TritonConfig]: ...
    def call_HOP(
        self, variable, grids, combined_args: dict[str, Any], tx
    ) -> ConstantVariable | None: ...
    def check_grid(
        self, grid
    ) -> tuple[int | sympy.Expr | SymInt, ...] | tuple[Proxy, ...]: ...
    def init_variable(
        self,
        variable: TraceableTritonKernelWrapper | TritonKernelVariable,
        kernel: TritonKernelType,
        kernel_idx: int | None,
        grid: TritonGridType | None,
    ) -> None: ...
    def call_getitem(
        self,
        variable: TritonKernelVariable | TraceableTritonKernelWrapper,
        args: Sequence[Any],
    ) -> TritonKernelVariable | TraceableTritonKernelWrapper: ...
    def call_run(
        self,
        variable: TritonKernelVariable | TraceableTritonKernelWrapper,
        args: Sequence[Any],
        kwargs: dict[str, Any],
        tx: InstructionTranslator | None,
    ) -> ConstantVariable | None: ...
    def call_triton_kernel(
        self,
        variable: TritonKernelVariable | TraceableTritonKernelWrapper,
        args: Sequence[Any],
        kwargs: dict[str, Any],
        tx: InstructionTranslator | None,
    ) -> ConstantVariable | None: ...

class TracingTritonHOPifier(TritonHOPifier):
    def raise_unsupported(self, msg: str) -> Never: ...
    def is_callable(self, maybe_callable: Any) -> bool: ...
    def get_value(self, val: Any) -> Any: ...
    def call_grid(
        self, grid: TritonGridCallableType, meta: TritonMetaParamsType, tx: None
    ) -> tuple[int | sympy.Expr | SymInt, ...]: ...
    def wrap_user_defined_obj(
        self,
        user_obj: Any,
        tx: InstructionTranslator | None,
        variable: TritonKernelVariable | TraceableTritonKernelWrapper | None,
        name: str,
    ) -> Any: ...
    def call_user_defined_fn(
        self,
        user_fn: Callable[..., Any],
        args: list,
        kwargs: dict,
        tx: InstructionTranslator | None,
        variable: TritonKernelVariable | TraceableTritonKernelWrapper | None,
    ) -> Any: ...
    def maybe_unpack_configs(
        self, configs: list[TritonConfig], tx: InstructionTranslator | None
    ) -> list[TritonConfig]: ...
    def maybe_unpack_heuristic_result(self, result: Any) -> Any: ...
    def check_grid(
        self, grid: TritonGridType
    ) -> tuple[int | sympy.Expr | SymInt, ...]: ...
    def store_non_graphable_args(
        self, combined_args: dict[str, Any]
    ) -> tuple[dict, int]: ...
    def call_HOP(
        self,
        variable: TraceableTritonKernelWrapper,
        grids: list[TritonGridTupleType],
        combined_args: dict[str, Any],
        tx: None,
    ) -> None: ...

tracing_triton_hopifier_singleton = ...

class TraceableTritonKernelWrapper:
    kernel: TritonKernelType
    kernel_idx: int | None
    grid: TritonGridType | None
    def __init__(
        self,
        kernel: TritonKernelType,
        kernel_idx: int | None,
        grid: TritonGridType | None,
    ) -> None: ...
    def __getitem__(self, *args: Sequence[Any]) -> TraceableTritonKernelWrapper: ...
    def run(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any: ...
    def __call__(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any: ...
    def specialize_symbolic(self, arg: Sequence[Any]) -> Any: ...
