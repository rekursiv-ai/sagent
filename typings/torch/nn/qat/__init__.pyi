from torch.nn.qat import (
    dynamic as dynamic,
    modules as modules,
)
from torch.nn.qat.modules import *

r"""QAT Dynamic Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.dynamic` instead.
"""
__all__ = ["Conv1d", "Conv2d", "Conv3d", "Embedding", "EmbeddingBag", "Linear"]
