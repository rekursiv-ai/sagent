from dataclasses import dataclass

import torch

__all__: list[str] = ...
_FUNCTIONAL_OPTIM_STEP_METHOD_NAME = ...

class _OptimizerHookState:
    __slots__ = ...
    def __init__(self, functional_optim, params=...) -> None: ...

@dataclass
class _OptimInBackwardHookState:
    optim_stream: torch.Stream
    wait_for_optim_stream_enqueued: bool
