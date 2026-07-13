from collections.abc import Callable, Sequence
from typing import IO, Any

from torch import fx, nn
from torch._inductor.compile_fx import _CompileFxCallable

import torch

"""
Utilities for reproducing and debugging issues in PyTorch's Dynamo AOT compilation.

This module provides tools and infrastructure for:
1. Generating minimal reproducible test cases ("repros") from failing compilations
2. Analyzing accuracy issues between eager and compiled execution
3. Minifying large models/inputs to isolate problematic patterns
4. Debugging compiler errors and accuracy divergences

The main components include:
- Repro generation: Creates standalone Python files that reproduce compiler issues
- Minification: Reduces large graphs to minimal failing examples
- Accuracy analysis: Compares compiled vs eager execution, with fp64 reference
- Debug tools: Dumps graph state, tracks intermediates, analyzes divergences

This is primarily used by PyTorch developers and researchers to debug issues in
the Dynamo AOT compilation pipeline, particularly for the Inductor backend.
"""
log = ...
inductor_config = ...
use_buck = ...

def wrap_compiler_debug(
    unconfigured_compiler_fn: _CompileFxCallable, compiler_name: str
) -> _CompileFxCallable: ...
def maybe_fbcode_instructions() -> str: ...
def generate_compiler_repro_string(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    *,
    stable_output: bool = ...,
    save_dir: str | None = ...,
    stable_hash: bool = ...,
    has_distributed_ops: bool = ...,
) -> str: ...
def save_graph_repro(
    fd: IO[Any],
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    *,
    stable_output: bool = ...,
    save_dir: str | None = ...,
    command: str = ...,
    accuracy: str | bool | None = ...,
    tracing_mode: str | None = ...,
    check_str: str | None = ...,
    stable_hash: bool = ...,
) -> None: ...
def dump_compiler_graph_state(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    *,
    accuracy: str | bool | None = ...,
) -> None: ...
def dump_to_minify(
    gm: torch.fx.GraphModule, args: Sequence[Any], compiler_name: str
) -> None: ...
def isolate_fails(
    fx_g: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    env: dict[str, Any] | None = ...,
    save_dir: str | None = ...,
    accuracy: bool | str | None = ...,
    tracing_mode: str | None = ...,
    check_str: str | None = ...,
) -> bool: ...
def inductor_fails(
    fx_g: torch.fx.GraphModule, args: Sequence[Any], check_str: str | None = ...
) -> bool: ...
def inductor_accuracy_fails(
    fx_g: torch.fx.GraphModule,
    args: Sequence[Any],
    check_str: str | None = ...,
    *,
    require_fp64: bool = ...,
    ignore_non_fp: bool = ...,
) -> bool: ...

backend_aot_accuracy_fails = ...

def repro_common(
    options: Any, mod: nn.Module, load_args: Any
) -> tuple[torch.fx.GraphModule, Sequence[Any]]: ...

ACCURACY_FAILS: dict[str, Callable[[torch.fx.GraphModule, Any], bool]] = ...

def repro_minifier_query(options: Any, mod: nn.Module, load_args: Any) -> None: ...
def repro_minify(options: Any, mod: nn.Module, load_args: Any) -> None: ...
def repro_analyze(options: Any, mod: nn.Module, load_args: Any) -> None:
    class WriterInterp(fx.Interpreter): ...
    class ExactReaderInterp(fx.Interpreter): ...
    class ReaderInterp(fx.Interpreter): ...

def repro_get_args(
    options: Any, mod: nn.Module, load_args: Any
) -> tuple[torch.fx.GraphModule, list[Any]]: ...
def repro_run(options: Any, mod: nn.Module, load_args: Any) -> None: ...
def run_repro(
    mod: nn.Module,
    load_args: Any,
    *,
    command: str = ...,
    accuracy: bool | str = ...,
    save_dir: str | None = ...,
    tracing_mode: str | None = ...,
    patch_code: str | None = ...,
    check_str: str | None = ...,
    **kwargs: Any,
) -> Any: ...
