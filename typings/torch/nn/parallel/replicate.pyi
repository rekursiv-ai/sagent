from collections.abc import Sequence
from typing import TypeVar

from torch.nn.modules import Module

import torch

__all__ = ["replicate"]
T = TypeVar("T", bound=Module)

def replicate(
    network: T, devices: Sequence[int | torch.device], detach: bool = ...
) -> list[T]: ...
