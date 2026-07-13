from contextlib import contextmanager

from torch import nn

import torch

"""
Since, https://github.com/huggingface/transformers/pull/36963, loading is always performed with models on meta
device. But since the `init_empty_weights` and `find_tied_parameters` functions are from accelerate, and accelerate is
somewhat still a soft dependency, we copy the functions here to be used natively in Transformers.

The `init_empty_weights` and `init_on_device` functions were copied from `accelerate.big_modeling.py`, and the
`find_tied_parameters` was copied from `accelerate.utils.modeling.py`
"""
logger = ...

@contextmanager
def init_empty_weights(include_buffers: bool = ...):  # -> Generator[None, Any, None]:
    ...
@contextmanager
def init_on_device(
    device: torch.device, include_buffers: bool = ...
):  # -> Generator[None, Any, None]:
    ...
def find_tied_parameters(model: nn.Module, **kwargs):  # -> list[list[Any]]:
    ...
