from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, override

import contextlib
import functools

from torch._dynamo.precompile_context import PrecompileCacheArtifact
from torch._inductor.codecache import (
    FxGraphCachePickler,
    FxGraphHashDetails,
    GuardedCache,
)
from torch._inductor.compile_fx import _CompileFxKwargs
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.output_code import CompiledFxGraph, OutputCode
from torch._inductor.remote_cache import JsonDataTy, RemoteCache
from torch._inductor.utils import BoxedBool
from torch.compiler._cache import CacheArtifact, CacheArtifactFactory
from torch.fx.node import Node

import torch

from .runtime_wrappers import CompilerWrapper, SubclassMeta
from .schemas import AOTAutogradCacheInfo, AOTConfig, ViewAndMutationMeta

"""
Utils for caching the outputs of AOTAutograd
"""
log = ...

class BypassAOTAutogradCache(Exception): ...
class FXGraphCacheMiss(BypassAOTAutogradCache): ...

def should_use_remote_autograd_cache() -> bool: ...
def should_use_local_autograd_cache() -> bool: ...
def should_bundle_autograd_cache() -> bool: ...
def check_node_safe(node: Node) -> None: ...
def check_cacheable(gm: torch.fx.GraphModule) -> None: ...

class AOTAutogradCacheDetails(FxGraphHashDetails):
    def get_triton_source_codes_from_gm(
        self, gm: torch.fx.GraphModule
    ) -> list[Any]: ...
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs,
        aot_config: AOTConfig,
        fx_config: _CompileFxKwargs,
    ) -> None: ...

class AOTAutogradCachePickler(FxGraphCachePickler):
    def __init__(self, gm: torch.fx.GraphModule) -> None: ...

@contextlib.contextmanager
def normalize_placeholder_names(
    gm: torch.fx.GraphModule,
) -> Generator[None, Any, None]: ...
def autograd_cache_key(
    gm: torch.fx.GraphModule,
    example_inputs,
    config: AOTConfig,
    fx_config: _CompileFxKwargs,
) -> tuple[str, list[str]]: ...

TOut = TypeVar("TOut", bound=OutputCode)

class InductorOutput(ABC, Generic[TOut]):
    @abstractmethod
    def pre_save(self) -> None: ...
    @abstractmethod
    def load(self, example_inputs) -> TOut: ...
    @abstractmethod
    def post_compile(self, result: TOut, fx_config: _CompileFxKwargs) -> TOut: ...

@dataclass
class CompiledFxGraphLoadable(InductorOutput[CompiledFxGraph]):
    result: CompiledFxGraph
    def pre_save(self) -> None: ...
    def load(self, example_inputs) -> CompiledFxGraph: ...
    def post_compile(
        self, result: CompiledFxGraph, fx_config: _CompileFxKwargs
    ) -> CompiledFxGraph: ...

@dataclass
class FxGraphCacheLoadable(InductorOutput[CompiledFxGraph]):
    fx_graph_cache_info: tuple[str, list[str]]
    fx_graph_guard_expr: str | None
    def pre_save(self) -> None: ...
    def load(self, example_inputs) -> CompiledFxGraph: ...
    def post_compile(
        self, result: CompiledFxGraph, fx_config: _CompileFxKwargs
    ) -> CompiledFxGraph: ...

@dataclass
class CompiledForward(FxGraphCacheLoadable): ...

@dataclass
class GenericCompiledBackward(InductorOutput[TOut]):
    backward_state_indices: list[int]
    num_symints_saved_for_bw_: int

@dataclass
class CompiledBackward(GenericCompiledBackward[CompiledFxGraph], FxGraphCacheLoadable):
    def post_compile(
        self, result: CompiledFxGraph, fx_config: _CompileFxKwargs
    ) -> CompiledFxGraph: ...

class BundledCompiledForward(CompiledFxGraphLoadable): ...

@dataclass
class BundledCompiledBackward(
    GenericCompiledBackward[CompiledFxGraph], CompiledFxGraphLoadable
):
    def post_compile(
        self, result: CompiledFxGraph, fx_config: _CompileFxKwargs
    ) -> CompiledFxGraph: ...

@dataclass
class SerializedGraphModule:
    fn: Callable[[dict[Any, Any], str], torch.nn.Module]
    args: tuple[Any, ...]
    def __init__(self, gm: torch.fx.GraphModule) -> None: ...
    def deserialize(self) -> torch.fx.GraphModule: ...

def serialize_graph_module(gm: torch.fx.GraphModule) -> SerializedGraphModule: ...

TForward = TypeVar("TForward", bound=InductorOutput)
TBackward = TypeVar("TBackward", bound=GenericCompiledBackward)

@dataclass
class GenericAOTAutogradCacheEntry(Generic[TForward, TBackward]):
    compiled_fw: TForward
    compiled_bw: TBackward | None
    aot_joint_graph_str: str | None
    aot_forward_graph_str: str | None
    aot_backward_graph_str: str | None
    runtime_metadata: ViewAndMutationMeta
    dispatch_wrappers: list[CompilerWrapper]
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None
    indices_of_inps_to_detach: list[int]
    forward_time_taken_ns: int
    backward_time_taken_ns: int
    sanitized_aot_config: AOTConfig
    guards_expr: str | None
    serialized_bw_module: SerializedGraphModule | None
    def pre_save(self) -> None: ...
    def wrap_post_compile(
        self,
        args: list[torch.Tensor],
        aot_config: AOTConfig,
        fx_config: _CompileFxKwargs,
    ) -> Callable: ...

class AOTAutogradCacheEntry(
    GenericAOTAutogradCacheEntry[CompiledForward, CompiledBackward]
): ...
class BundledAOTAutogradCacheEntry(
    GenericAOTAutogradCacheEntry[BundledCompiledForward, BundledCompiledBackward]
): ...

@contextlib.contextmanager
def sanitize_gm_for_cache(gm: torch.fx.GraphModule) -> Generator[None, Any, None]: ...

@CacheArtifactFactory.register
class AOTAutogradCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None: ...
    @override
    @staticmethod
    def type() -> Literal[aot_autograd]: ...

@CacheArtifactFactory.register
class BundledAOTAutogradCacheArtifact(PrecompileCacheArtifact[Callable]):
    @override
    @staticmethod
    def type() -> Literal[precompile_aot_autograd]: ...
    @override
    def after_deserialization(self) -> Callable: ...

class AOTAutogradCache(GuardedCache[GenericAOTAutogradCacheEntry]):
    @staticmethod
    def clear() -> None: ...
    @staticmethod
    def try_load(
        mod: torch.fx.GraphModule | torch._dynamo.utils.GmWrapper,
        args,
        aot_config: AOTConfig,
        cudagraphs: BoxedBool,
        boxed_forward_device_index: BoxedDeviceIndex | None,
        local: bool,
        remote: bool,
    ) -> Callable | None: ...
    @classmethod
    def generate_guards_expression(
        cls: type[AOTAutogradCache], cache_info: AOTAutogradCacheInfo
    ) -> str | None: ...
    @staticmethod
    def evaluate_guards(
        guard_expr: str, hints: list[int] | list[torch.SymInt]
    ) -> bool: ...
    @staticmethod
    def save(key: str, entry: GenericAOTAutogradCacheEntry, remote: bool) -> None: ...
    @staticmethod
    @functools.cache
    def get_remote_cache() -> RemoteCache[JsonDataTy] | None: ...
    @staticmethod
    def make_entry(
        compiled_fw_func: CompiledFxGraph,
        compiled_bw_func: CompiledFxGraph | None,
        aot_joint_graph_str: str | None,
        aot_forward_graph_str: str | None,
        aot_backward_graph_str: str | None,
        runtime_metadata: ViewAndMutationMeta,
        dispatch_wrappers: list[CompilerWrapper],
        maybe_subclass_meta: SubclassMeta | None,
        num_fw_outs_saved_for_bw: int | None,
        indices_of_inps_to_detach: list[int],
        forward_time_taken_ns: int,
        backward_time_taken_ns: int,
        sanitized_aot_config: AOTConfig,
        guards_expr: str | None,
        backward_state_indices: list[int] | None,
        num_symints_saved_for_bw: int | None,
        serialized_bw_module: SerializedGraphModule | None,
    ) -> GenericAOTAutogradCacheEntry: ...
