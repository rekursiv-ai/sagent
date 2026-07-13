from collections.abc import Sequence
from typing import IO, Any

from torch.export import ExportedProgram

import torch

"""
Utilities for debugging and reproducing issues in Ahead of Time with Inductor (AOTI) compilation.

This file provides tools and utilities for:
- Generating minimal reproducible test cases (minification)
- Handling exported programs and graph modules
- Creating debug repros for AOTI compilation issues
- Supporting both accuracy testing and error reproduction
- Managing configuration and environment for repro cases

The main components include:
- Minification tools to reduce test cases while preserving errors
- Repro generation utilities for exported programs
- Error handling specific to AOTI compilation
- Command-line interface for running and managing repros
"""
log = ...
inductor_config = ...
use_buck = ...

class AOTIMinifierError(Exception):
    def __init__(self, original_exception: str | Exception) -> None: ...

def dump_to_minify(
    exported_program: ExportedProgram,
    compiler_name: str,
    command: str = ...,
    options: dict[str, Any] | None = ...,
) -> None: ...
def get_module_string(gm: torch.fx.GraphModule) -> str: ...
def save_graph_repro_ep(
    fd: IO[Any],
    compiler_name: str,
    *,
    exported_program: ExportedProgram | None = ...,
    gm: torch.nn.Module | None = ...,
    args: tuple[Any] | None = ...,
    config_patches: dict[str, str] | None = ...,
    stable_output: bool = ...,
    save_dir: str | None = ...,
    command: str = ...,
    accuracy: str | bool | None = ...,
    check_str: str | None = ...,
    module_in_comment: bool = ...,
    strict: bool = ...,
) -> None: ...
def dump_compiler_graph_state(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    *,
    config_patches: dict[str, str] | None = ...,
    accuracy: str | bool | None = ...,
    strict: bool = ...,
) -> None: ...
def generate_compiler_repro_exported_program(
    exported_program: ExportedProgram,
    *,
    options: dict[str, str] | None = ...,
    stable_output: bool = ...,
    save_dir: str | None = ...,
) -> str: ...
def repro_load_args(load_args: Any, save_dir: str | None) -> tuple[Any]: ...
def repro_common(
    options: Any, exported_program: ExportedProgram
) -> tuple[torch.fx.GraphModule, Any, Any]: ...
def repro_get_args(
    options: Any,
    exported_program: ExportedProgram,
    config_patches: dict[str, Any] | None,
) -> tuple[torch.fx.GraphModule, Any, Any]: ...
def repro_run(
    options: Any,
    exported_program: ExportedProgram,
    config_patches: dict[str, Any] | None,
) -> None: ...
def export_for_aoti_minifier(
    gm: torch.nn.Module,
    tuple_inputs: tuple[Any],
    strict: bool = ...,
    skip_export_error: bool = ...,
) -> torch.nn.Module | None: ...
def repro_minify(
    options: Any,
    exported_program: ExportedProgram,
    config_patches: dict[str, Any] | None,
) -> None: ...
def run_repro(
    exported_program: ExportedProgram,
    *,
    config_patches: dict[str, str] | None = ...,
    command: str = ...,
    accuracy: bool | str = ...,
    save_dir: str | None = ...,
    tracing_mode: str | None = ...,
    check_str: str | None = ...,
    minifier_export_mode: str = ...,
    skip_export_error: bool = ...,
    **more_kwargs: Any,
) -> Any: ...
