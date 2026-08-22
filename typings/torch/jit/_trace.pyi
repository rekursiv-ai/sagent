from collections.abc import Callable
from enum import StrEnum
from typing import Any, LiteralString, TypeVar
from typing_extensions import ParamSpec

from torch import Tensor
from torch.jit._script import ScriptModule

import torch

"""Tracing.

This module contains functionality to support the JIT's tracing frontend, notably:
    * torch.jit.trace
    * torch.jit.trace_module

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
_flatten = ...
_unflatten = ...
R = TypeVar("R", covariant=True)
P = ParamSpec("P")

class ONNXTracedModule(torch.nn.Module):
    def __init__(
        self,
        inner,
        strict=...,
        force_outplace=...,
        return_inputs=...,
        return_inputs_states=...,
    ) -> None: ...
    def forward(
        self, *args: torch.Tensor
    ) -> tuple[Graph, Any, Any] | tuple[Graph, Any]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Graph, Any, Any] | tuple[Graph, Any]: ...

_JIT_TIME = ...
_JIT_DISABLE = ...
_JIT_STATS = ...

def verify(model, args, loss_fn=..., devices=...) -> None: ...
def indent(s) -> LiteralString: ...

class TracingCheckError(Exception):
    def __init__(
        self, graph_diff_error, tensor_compare_error, extra_msg=...
    ) -> None: ...

class TracerWarning(Warning):
    @staticmethod
    def ignore_lib_warnings() -> None: ...

def make_tuple(example_inputs) -> tuple[Tensor | dict[Any, Any]] | tuple[Any, ...]: ...
def make_module(
    mod, _module_class, _compilation_unit
) -> ScriptModule | TopLevelTracedModule: ...
def wrap_check_inputs(check_inputs) -> list[dict[str, Any]] | None: ...
def analyze_ts_result_with_export_result(export, trace) -> bool: ...

class _ExportType(StrEnum):
    DIRECT_EXPORT = ...
    TRACE_AND_EXPORT = ...
    SOURCE_TO_SOURCE = ...

class _ExportOutcome(StrEnum):
    SUCCESS = ...
    FAILED_TO_EXPORT = ...
    FAILED_TO_RUN = ...
    ACCURACY_ERROR = ...

def trace(
    func,
    example_inputs=...,
    optimize=...,
    check_trace=...,
    check_inputs=...,
    check_tolerance=...,
    strict=...,
    _force_outplace=...,
    _module_class=...,
    _compilation_unit=...,
    example_kwarg_inputs=...,
    _store_inputs=...,
) -> ScriptModule | TopLevelTracedModule | tuple[Graph, Stack]: ...

_trace_module_map: dict[Any, Any] | None = ...

def trace_module(
    mod,
    inputs,
    optimize=...,
    check_trace=...,
    check_inputs=...,
    check_tolerance=...,
    strict=...,
    _force_outplace=...,
    _module_class=...,
    _compilation_unit=...,
    example_inputs_is_kwarg=...,
    _store_inputs=...,
) -> ScriptModule | TopLevelTracedModule: ...
def is_tracing() -> bool: ...

class TracedModule(ScriptModule):
    _disable_script_meta = ...
    def __init__(self, orig, id_set=..., _compilation_unit=...) -> None: ...
    def forward(self, *args, **kwargs): ...
    def __getattr__(self, attr) -> Any: ...
    def __setattr__(self, attr, value) -> None: ...
    def extra_repr(self) -> str: ...

class TopLevelTracedModule(TracedModule):
    forward: Callable[..., Any] = ...
