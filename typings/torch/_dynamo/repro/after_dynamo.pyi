from collections.abc import Sequence
from typing import Any

from torch import fx

import torch

from ..backends.registry import CompilerFn, register_debug_backend

"""
Utilities for reproducing and debugging issues in Dynamo after graph capture.

This file provides tools and infrastructure for debugging problems that occur
after Dynamo has captured the graph but before/during backend compilation.
Key components include:

- Minification tools to reduce large graphs to minimal failing examples
- Accuracy testing to validate compiled graph outputs match eager mode
- Repro generation to create standalone reproduction scripts
- Debug backends for capturing and analyzing failures
- Utilities for saving/loading graph states and inputs

The tools here focus specifically on the post-graph-capture stage, making them
useful for debugging backend compilation issues, AOTAutograd problems, and
accuracy discrepancies between compiled and eager execution.
"""
log = ...
inductor_config = ...
use_buck = ...

class WrapBackendDebug:
    def __init__(
        self, unconfigured_compiler_fn: CompilerFn, compiler_name: str | None
    ) -> None: ...
    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[Any], **kwargs: Any
    ) -> torch.fx.GraphModule: ...

def wrap_backend_debug(
    unconfigured_compiler_fn: CompilerFn, compiler_name: str | None
) -> WrapBackendDebug: ...
def generate_dynamo_fx_repro_string(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str | None,
    check_accuracy: bool = ...,
    *,
    stable_output: bool = ...,
    save_dir: str | None = ...,
    command: str = ...,
) -> str: ...
def dump_backend_repro_as_file(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str | None,
    check_accuracy: bool = ...,
) -> None: ...
def dump_backend_state(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str | None,
    check_accuracy: bool = ...,
) -> None: ...
def dump_to_minify_after_dynamo(
    gm: torch.fx.GraphModule, args: Sequence[Any], compiler_name: str | None
) -> None: ...
@register_debug_backend
def dynamo_minifier_backend(
    gm: fx.GraphModule, example_inputs: Sequence[Any], compiler_name: str | None
) -> fx.GraphModule: ...
@register_debug_backend
def dynamo_accuracy_minifier_backend(
    gm: fx.GraphModule, example_inputs: Sequence[Any], compiler_name: str | None
) -> fx.GraphModule: ...
def backend_fails(
    gm: fx.GraphModule,
    example_inputs: Sequence[Any],
    compiler_fn: CompilerFn,
    orig_failure: Sequence[Any],
) -> bool: ...
def run_load_args(options: Any, mod: torch.nn.Module, load_args: Any) -> list[Any]: ...
def repro_minify(options: Any, mod: torch.nn.Module, load_args: Any) -> None: ...
def repro_run(options: Any, mod: torch.nn.Module, load_args: Any) -> None: ...
def run_repro(
    mod: torch.nn.Module,
    load_args: Any,
    *,
    command: str = ...,
    accuracy: bool | str = ...,
    save_dir: str | None = ...,
    autocast: bool = ...,
    backend: str = ...,
    **kwargs: Any,
) -> None: ...
