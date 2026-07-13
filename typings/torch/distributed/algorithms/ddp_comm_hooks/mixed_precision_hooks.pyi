from dataclasses import dataclass
from typing import Any

import torch

@dataclass
class _AllreduceUpcastHookState:
    ddp_weakref: Any
    upcast_stream: torch.Stream
    wait_for_stream_enqueued: bool = ...
