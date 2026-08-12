from collections.abc import Callable, Sequence
from torch import Tensor
from typing import (
    IO as IO,
    TYPE_CHECKING as TYPE_CHECKING,
    Any as Any,
    Literal as Literal,
    Optional as Optional,
    Union as Union,
)

from torch._inductor.utils import InputType as InputType
from torch.export import ExportedProgram as ExportedProgram
from torch.export.pt2_archive._package import AOTICompiledModel as AOTICompiledModel
from torch.export.pt2_archive._package_weights import Weights as Weights
from torch.types import FileLike as FileLike

import torch.fx

from .standalone_compile import CompiledArtifact as CompiledArtifact

__all__ = [
    "compile",
    "cudagraph_mark_step_begin",
    "list_mode_options",
    "list_options",
    "standalone_compile",
]
log = ...

def compile(
    gm: torch.fx.GraphModule,
    example_inputs: list[InputType],
    options: dict[str, Any] | None = ...,
) -> Callable[[list[object]], Sequence[Tensor]] | str | list[str] | Weights: ...
def aoti_compile_and_package(
    exported_program: ExportedProgram,
    _deprecated_unused_args=...,
    _deprecated_unused_kwargs=...,
    *,
    package_path: FileLike | None = ...,
    inductor_configs: dict[str, Any] | None = ...,
) -> str: ...
def aoti_load_package(
    path: FileLike, run_single_threaded: bool = ..., device_index: int = ...
) -> AOTICompiledModel: ...
def aot_compile(
    gm: torch.fx.GraphModule,
    args: tuple[Any],
    kwargs: dict[str, Any] | None = ...,
    *,
    options: dict[str, Any] | None = ...,
) -> str | list[str | Weights] | torch.fx.GraphModule: ...
def list_mode_options(
    mode: str | None = ..., dynamic: bool | None = ...
) -> dict[str, Any]: ...
def list_options() -> list[str]: ...
def cudagraph_mark_step_begin() -> None: ...
def standalone_compile(
    gm: torch.fx.GraphModule,
    example_inputs: list[InputType],
    *,
    dynamic_shapes: Literal[
        "from_example_inputs", "from_tracing_context", "from_graph"
    ] = ...,
    options: dict[str, Any] | None = ...,
) -> CompiledArtifact: ...
