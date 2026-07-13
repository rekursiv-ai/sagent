from typing import TYPE_CHECKING as TYPE_CHECKING

from torch._C._monitor import *
from torch._C._monitor import (
    _WaitCounter as _WaitCounter,
    _WaitCounterTracker as _WaitCounterTracker,
)
from torch.utils.tensorboard import SummaryWriter as SummaryWriter

STAT_EVENT = ...

class TensorboardEventHandler:
    def __init__(self, writer: SummaryWriter) -> None: ...
    def __call__(self, event: Event) -> None: ...
