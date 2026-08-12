from typing import Literal
from torch.distributed._shard.sharded_tensor import _sharded_op_impl

import torch

@_sharded_op_impl(torch._has_compatible_shallow_copy_type)
def tensor_has_compatible_shallow_copy_type(
    types, args=..., kwargs=..., pg=...
) -> Literal[False]: ...
