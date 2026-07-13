from typing import Any

from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .schemas import AOTConfig, FakifiedFlatArgs

static_inputs_log = ...

def process_inputs(
    flat_args: list[Any],
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: ShapeEnv | None,
    ignore_shape_env: bool = ...,
) -> FakifiedFlatArgs: ...
def construct_fake_mode(
    flat_args: list[Any], aot_config: AOTConfig
) -> tuple[FakeTensorMode, ShapeEnv | None]: ...
