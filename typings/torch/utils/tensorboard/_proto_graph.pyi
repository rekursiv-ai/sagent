from collections.abc import Sequence

from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto

import torch

def attr_value_proto(
    dtype: object, shape: Sequence[int] | None, s: str | None
) -> dict[str, AttrValue]: ...
def tensor_shape_proto(outputsize: Sequence[int]) -> TensorShapeProto: ...
def node_proto(
    name: str,
    op: str = ...,
    input: list[str] | str | None = ...,
    dtype: torch.dtype | None = ...,
    shape: tuple[int, ...] | None = ...,
    outputsize: Sequence[int] | None = ...,
    attributes: str = ...,
) -> NodeDef: ...
