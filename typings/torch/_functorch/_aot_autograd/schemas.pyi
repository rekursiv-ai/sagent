from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    NewType,
    Protocol,
    TypeVar,
)

import contextlib
import functools

from torch import SymInt, Tensor
from torch._guards import Source
from torch._inductor.output_code import OutputCode
from torch._inductor.utils import InputType
from torch._ops import OpOverload
from torch.fx.experimental._backward_state import BackwardState

import torch
import torch.utils._pytree as pytree

from .descriptors import AOTInput, AOTOutput
from .functional_utils import ViewMetaSequence
from .graph_capture_wrappers import JointFnHandle

"""
The various dataclasses, Enums, namedtuples etc used in AOTAutograd. This includes
input/output types, metadata, config, function signatures etc.
"""
zip = ...
OutputType = ...

@dataclass(frozen=True)
class OutputAliasInfo:
    output_type: OutputType
    raw_type: type
    base_idx: int | None
    dynamic_dims: set[int] | None
    requires_grad: bool
    view_meta_sequence: ViewMetaSequence | None = ...

class MutationType(Enum):
    NOT_MUTATED = ...
    MUTATED_IN_GRAPH = ...
    MUTATED_OUT_GRAPH = ...

@dataclass(frozen=True)
class InputAliasInfo:
    is_leaf: bool
    mutates_data: bool
    mutates_metadata: bool
    mutations_hidden_from_autograd: bool
    mutations_under_no_grad_or_inference_mode: bool
    mutation_inductor_storage_resize: bool
    mutates_storage_metadata: bool
    requires_grad: bool
    keep_input_mutations: bool
    def __post_init__(self) -> None: ...
    @functools.cached_property
    def mutation_type(self) -> MutationType: ...

@dataclass
class MemoryFormatMeta:
    size: Sequence[int] | None = ...
    stride: Sequence[int] | None = ...
    memory_format: torch.memory_format | None = ...
    @staticmethod
    def from_tensor(t: torch.Tensor) -> MemoryFormatMeta | None: ...

@dataclass
class PlainTensorMeta:
    unwrapped_idx: int
    memory_format: MemoryFormatMeta | None = ...

@dataclass
class SubclassCreationMeta:
    flat_tensor_start_idx: int
    arg_count: int
    included_subclass_symints: bool
    attrs: dict[str, SubclassCreationMeta | PlainTensorMeta]
    outer_size: Iterable[int | torch.SymInt | None]
    outer_stride: Iterable[int | torch.SymInt | None]
    meta: Any
    original_subclass: torch.Tensor | None
    original_subclass_type: type | None = ...
    memory_format: MemoryFormatMeta | None = ...
    def compute_outer_size_and_stride(
        self, all_args, *, curr_start_idx: int
    ) -> tuple[PyTree | Any, PyTree | Any]: ...
    def creation_fn(self, all_args, *, is_runtime: bool): ...
    def make_runtime_safe(self) -> None: ...
    def __post_init__(self) -> None: ...

@dataclass(eq=False)
class ViewAndMutationMeta:
    input_info: list[InputAliasInfo]
    output_info: list[OutputAliasInfo]
    num_intermediate_bases: int
    keep_input_mutations: bool
    traced_tangents: list[Any]
    traced_tangents_descs: list[AOTInput]
    subclass_inp_meta: list[PlainTensorMeta | SubclassCreationMeta]
    subclass_fw_graph_out_meta: list[PlainTensorMeta | SubclassCreationMeta]
    subclass_tangent_meta: list[PlainTensorMeta | SubclassCreationMeta]
    is_train: bool = ...
    traced_tangent_metas: list[Any] | None = ...
    num_symints_saved_for_bw: int | None = ...
    grad_enabled_mutation: bool | None = ...
    deterministic: bool | None = ...
    static_input_indices: list[int] = ...
    tokens: dict[Any, torch.Tensor] = ...
    indices_of_inputs_that_requires_grad_with_mutations_in_bw: list[int] = ...
    bw_donated_idxs: list[int] | None = ...
    num_backward_tokens: int = ...
    num_graphsafe_rng_states: int = ...
    graphsafe_rng_state_index: int | None = ...
    def __post_init__(self) -> None: ...
    def make_runtime_safe(self) -> None: ...
    @property
    def tensors_saved_for_backwards_slice(
        self,
    ) -> slice[int, int, Any] | slice[int, Any, Any]: ...
    @property
    def symints_saved_for_backwards_slice(
        self,
    ) -> slice[int, Any, Any] | slice[int, int, Any]: ...
    def __eq__(self, other) -> bool: ...

@dataclass(eq=False)
class SubclassMeta:
    fw_metadata: ViewAndMutationMeta
    grad_input_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = ...
    def __init__(self) -> None: ...

@dataclass(frozen=True)
class TensorAlias:
    alias: torch.Tensor

@dataclass
class BackwardSignature:
    gradients_to_parameters: dict[str, str]
    gradients_to_user_inputs: dict[str, str]
    loss_output: str

GraphOutputName = NewType("GraphOutputName", str)
GraphInputName = NewType("GraphInputName", str)
FQN = NewType("FQN", str)

@dataclass
class GraphSignature:
    parameters: list[FQN]
    buffers: list[FQN]
    user_inputs: list[GraphInputName]
    user_outputs: list[GraphOutputName]
    inputs_to_parameters: dict[GraphInputName, FQN]
    inputs_to_buffers: dict[GraphInputName, FQN]
    buffers_to_mutate: dict[GraphOutputName, FQN]
    parameters_to_mutate: dict[GraphOutputName, FQN]
    user_inputs_to_mutate: dict[GraphOutputName, GraphInputName]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    backward_signature: BackwardSignature | None
    input_tokens: list[GraphInputName]
    output_tokens: list[GraphOutputName]
    @classmethod
    def from_tracing_metadata(
        cls,
        *,
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
        graph_input_names: list[str],
        graph_output_names: list[str],
        view_mutation_metadata: ViewAndMutationMeta,
        named_parameters: list[str],
        named_buffers: list[str],
        num_user_inputs: int,
        num_user_outputs: int,
        trace_joint: bool,
        loss_index: int | None,
        backward_signature: BackwardSignature | None,
    ) -> GraphSignature: ...

@dataclass
class AOTAutogradCacheInfo:
    cache_key: str
    start_time_ns: int
    forward_symints: list[torch.SymInt]

@dataclass
class AOTConfig:
    fw_compiler: Callable
    bw_compiler: Callable
    partition_fn: Callable
    decompositions: dict[OpOverload, Callable]
    num_params_buffers: int
    aot_id: int
    keep_inference_input_mutations: bool
    is_export: bool = ...
    no_tangents: bool = ...
    dynamic_shapes: bool = ...
    aot_autograd_arg_pos_to_source: list[Source] | None = ...
    static_input_indices: list[int] | None = ...
    inference_compiler: Callable | None = ...
    enable_log: bool = ...
    pre_dispatch: bool = ...
    cache_info: AOTAutogradCacheInfo | None = ...
    ignore_shape_env: bool = ...
    precompile_backend_id: str | None = ...
    force_non_lazy_backward_lowering: bool = ...
    export_trace_joint: bool = ...
    def __post_init__(self) -> None: ...

SubclassTracingInfo = ...

@dataclass
class AOTState:
    needs_autograd: bool
    flat_args: list[FxValue]
    flat_args_descs: list[AOTInput]
    fw_metadata: ViewAndMutationMeta
    aot_config: AOTConfig
    stack: contextlib.ExitStack

type FxValue = Tensor | int | SymInt | BackwardState

class CompilerWrapper:
    def pre_compile(
        self,
        flat_fn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[Callable, list[FxValue], list[AOTInput], ViewAndMutationMeta]: ...
    def post_compile(
        self, compiled_fn, aot_config, *, runtime_metadata
    ) -> Callable: ...

class InductorWrapper:
    def pre_compile(
        self,
        fw_module: torch.fx.GraphModule,
        flat_args: list[Tensor],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> None: ...
    def post_compile(
        self, compiled_fn, aot_config, *, runtime_metadata
    ) -> Callable: ...

@dataclass
class AOTGraphCapture:
    wrappers: list[CompilerWrapper]
    graph_module: torch.fx.GraphModule
    updated_flat_args: list[Any] | tuple[list[Any], list[Any]]
    updated_flat_args_descs: list[AOTInput] | tuple[list[AOTInput], list[AOTInput]]
    maybe_subclass_meta: Any

FakifiedFlatArgs = NewType("FakifiedFlatArgs", list[Any])
TOutputCode = TypeVar("TOutputCode", bound=OutputCode)

class AOTDispatchCompiler(Protocol):
    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[InputType]
    ) -> Any: ...

class SerializableAOTDispatchCompiler(AOTDispatchCompiler):
    def __init__(
        self,
        output_code_ty: type[TOutputCode],
        compiler_fn: Callable[[torch.fx.GraphModule, Sequence[InputType]], TOutputCode],
    ) -> None: ...
    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[InputType]
    ) -> OutputCode: ...

class FlatFn(Protocol):
    def __call__(self, *args: FxValue) -> list[FxValue]: ...

class TraceFn(Protocol):
    def __call__(self, *args: FxValue) -> tuple[list[FxValue], list[AOTOutput]]: ...

class PreppedForAutogradTraceFn(Protocol):
    def __call__(
        self, *args: FxValue
    ) -> tuple[tuple[list[FxValue], list[bool]], list[AOTOutput]]: ...

class JointTraceFn(Protocol):
    handle: JointFnHandle
    def __call__(
        self, primals: list[FxValue], tangents: list[FxValue]
    ) -> tuple[
        tuple[list[FxValue], list[Tensor | None]],
        tuple[list[AOTOutput], list[AOTOutput | None]],
    ]: ...

@dataclass
class JointWithDescriptors:
    _aot_state: AOTState
    _aot_graph_capture: AOTGraphCapture
    params_spec: list[str]
    buffers_spec: list[str]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    @property
    def graph_module(self) -> GraphModule: ...
    @graph_module.setter
    def graph_module(self, value) -> None: ...
