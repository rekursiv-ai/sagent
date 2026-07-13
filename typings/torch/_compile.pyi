from typing import TypeVar
from typing_extensions import ParamSpec

"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""
_T = TypeVar("_T")
_P = ParamSpec("_P")
