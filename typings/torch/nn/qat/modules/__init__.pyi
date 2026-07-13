from torch.ao.nn.qat.modules.conv import (
    Conv1d as Conv1d,
    Conv2d as Conv2d,
    Conv3d as Conv3d,
)
from torch.ao.nn.qat.modules.embedding_ops import (
    Embedding as Embedding,
    EmbeddingBag as EmbeddingBag,
)
from torch.ao.nn.qat.modules.linear import Linear as Linear
from torch.nn.qat.modules import (
    conv as conv,
    embedding_ops as embedding_ops,
    linear as linear,
)

r"""QAT Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.modules` instead.
"""
__all__ = ["Conv1d", "Conv2d", "Conv3d", "Embedding", "EmbeddingBag", "Linear"]
