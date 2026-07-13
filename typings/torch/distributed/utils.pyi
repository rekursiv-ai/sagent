from collections import OrderedDict
from typing import Any, TypeVar

from torch.nn.utils.rnn import PackedSequence

import torch

__all__ = []
S = TypeVar("S", dict, list, tuple)
T = TypeVar("T", torch.Tensor, PackedSequence)
Q = TypeVar("Q")
R = TypeVar("R", dict, list, tuple, set, OrderedDict, PackedSequence, Any)
