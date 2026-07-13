from collections.abc import Generator

from torch import nn

from .contract import _State, contract

class _CheckpointState(_State):
    enable_hook: bool = ...
    _ac_generator: Generator[None] | None

@contract(_CheckpointState)
def checkpoint(module: nn.Module, **kwargs) -> nn.Module: ...
