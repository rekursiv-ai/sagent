from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    Unpack,
    override,
)
from typing_extensions import ParamSpec, TypedDict

import contextlib
import enum
import types

from torch._inductor.cudagraph_utils import BoxedDeviceIndex, PlaceholderInfo
from torch._inductor.output_code import OutputCode
from torch._inductor.utils import BoxedBool, InputType
from torch._ops import OpOverload
from torch.export.pt2_archive._package_weights import Weights
from torch.fx import GraphModule

import torch.fx

from .ir import ExternKernelNode

_P = ParamSpec("_P")
_T = TypeVar("_T")
if TYPE_CHECKING or not config.is_fbcode():
    def time_and_log(attr: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
    def log_optimus_to_scuba(*args: object, **kwargs: object) -> None: ...

else: ...

class FxCompileMode(enum.Enum):
    NORMAL = ...
    SERIALIZE = ...
    SUBPROCESS = ...

@dataclass
class FxCompileConfig:
    mode: FxCompileMode
    use_async: bool
    use_progressive: bool

_fx_compile_config = ...
fx_compile_mode = ...
fx_compile_async = ...
fx_compile_progressive = ...
log = ...
perf_hint_log = ...
pre_grad_graphs_log = ...
post_grad_graphs_log = ...
static_inputs_log = ...
inductor_metrics_log = ...

def get_static_input_idxs(num_fixed: int) -> list[int]: ...
def record_original_output_strides(gm: GraphModule) -> None: ...
def split_const_gm(
    gm: GraphModule,
    skip_constructor: bool = ...,
    lifted_constant_names: list[str] | None = ...,
    skip_folding_node_fn: Callable[[torch.fx.Node], bool] | None = ...,
) -> tuple[GraphModule, dict[str, int]]: ...
def is_tf32_warning_applicable(gm: GraphModule) -> bool: ...
def maybe_disable_comprehensive_padding(
    example_inputs: Sequence[InputType],
) -> AbstractContextManager[None, None]: ...
def maybe_disable_graph_partition(
    cpp_wrapper: bool, aot_mode: bool
) -> AbstractContextManager[None, None]: ...
def fake_tensor_prop(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    force_allow_non_fake_inputs: bool = ...,
) -> torch._subclasses.FakeTensorMode: ...
def get_patched_config_dict(
    config_patches: str | dict[str, Any] | None = ...,
) -> dict[str, Any]: ...
@contextlib.contextmanager
def with_fresh_cache_if_config() -> Generator[None]: ...

class _CompileFxKwargs(TypedDict, total=False):
    cudagraphs: BoxedBool | None
    static_input_idxs: Sequence[int]
    is_backward: bool
    graph_id: int | None
    cpp_wrapper: bool
    aot_mode: bool
    is_inference: bool
    layout_opt: bool | None
    extern_node_serializer: Callable[[list[ExternKernelNode]], Any] | None
    boxed_forward_device_index: BoxedDeviceIndex | None
    fx_wrapper: bool

class _CompileFxCallable(Protocol):
    def __call__(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        **kwargs: Unpack[_CompileFxKwargs],
    ) -> OutputCode: ...

def compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    **kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode: ...

class _FxCompileStat:
    codegen_and_compile: int = ...

class FxCompile(ABC):
    _compile_stats: dict[type[FxCompile], _FxCompileStat] = ...
    @abstractmethod
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode: ...

class _InProcessFxCompile(FxCompile):
    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode: ...

def fx_codegen_and_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    inputs_to_check: Sequence[int],
    **graph_kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode: ...
def get_input_idxs_to_check(
    inputs: Sequence[InputType], static_input_idxs: Sequence[int]
) -> Sequence[int]: ...
def cudagraphify(
    model: Callable[..., Any],
    static_input_idxs: Sequence[int] = ...,
    *,
    device_index: int,
    stack_traces: list[str | None],
    is_backward: bool,
    is_inference: bool,
    constants: tuple[torch.Tensor, ...] = ...,
    placeholders: Sequence[PlaceholderInfo] = ...,
    mutated_input_idxs: tuple[int, ...] = ...,
) -> Callable[..., Any]: ...
def static_input(x: torch.Tensor) -> torch.Tensor: ...
def index_expanded_dims_and_copy_(
    dst: torch.Tensor, src: torch.Tensor, expanded_dims: list[int]
) -> None: ...
def cudagraphify_impl(
    model: Callable[..., Any],
    inputs: list[torch.Tensor],
    static_input_idxs: Sequence[int] = ...,
) -> Callable[[list[InputType]], Any]: ...
def compile_fx_aot(
    model_: GraphModule,
    example_inputs_: list[InputType],
    inner_compile: _CompileFxCallable = ...,
    config_patches: dict[str, Any] | None = ...,
) -> list[str | Weights] | str | GraphModule: ...

_graph_counter = ...

def fw_compiler_freezing(
    aot_autograd_model: GraphModule,
    aot_example_inputs: Sequence[InputType],
    dynamo_model: GraphModule,
    num_example_inputs: int,
    inner_compile: Callable[..., Any],
    cudagraphs: BoxedBool,
    graph_id: int,
    forward_device: BoxedDeviceIndex,
) -> Callable[[list[object]], Sequence[torch.Tensor]]: ...
def get_cpp_wrapper_config() -> dict[str, object]: ...
def get_cuda_device_context(
    gm: torch.fx.GraphModule,
) -> AbstractContextManager[None]: ...
def partition_fn(
    gm: GraphModule, joint_inputs: Sequence[object], **kwargs: object
) -> tuple[GraphModule, GraphModule]: ...
def get_num_model_outputs(model: GraphModule) -> int: ...

@dataclass(frozen=True)
class CompilerConfigExtra:
    cudagraphs: BoxedBool
    graph_id: int
    forward_device: BoxedDeviceIndex

def create_compiler_config_extra(config: types.ModuleType) -> CompilerConfigExtra: ...
def compile_fx_forward(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    num_orig_model_outputs: int,
    num_example_inputs: int,
    compiler_config_extra: CompilerConfigExtra,
    inner_compile: Callable[..., OutputCode] = ...,
    is_inference: bool = ...,
) -> OutputCode: ...
def compile_fx_backward(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    compiler_config_extra: CompilerConfigExtra,
    inner_compile: Callable[..., OutputCode] = ...,
) -> OutputCode: ...
def run_pre_grad_passes(
    model_: GraphModule, example_inputs_: Sequence[InputType]
) -> GraphModule: ...
def compile_fx(
    model_: GraphModule,
    example_inputs_: Sequence[InputType],
    inner_compile: Callable[..., OutputCode] = ...,
    config_patches: dict[str, Any] | None = ...,
    decompositions: dict[OpOverload, Callable[..., Any]] | None = ...,
    ignore_shape_env: bool = ...,
) -> Callable[[list[object]], Sequence[torch.Tensor]] | str | list[str] | Weights: ...
def graph_returns_tuple(gm: GraphModule) -> bool: ...
def make_graph_return_tuple(
    gm: GraphModule, inputs: Sequence[InputType], compile_gm: Callable[..., Any]
) -> Callable[..., Any]: ...
def handle_dynamo_export_graph(
    gm: GraphModule, inputs: Sequence[InputType], compile_gm: Callable[..., Any]
) -> Callable[..., Any]: ...
