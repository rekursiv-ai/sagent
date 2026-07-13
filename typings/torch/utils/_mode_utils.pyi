from typing import TypeVar

import torch

T = TypeVar("T")

def all_same_mode(modes) -> bool: ...

no_dispatch = torch._C._DisableTorchDispatch
