from collections.abc import Sequence

import dataclasses

import torch

"""Implementation of symbolic FX ops to represent arbitrary ONNX ops.

This module provides a way to create symbolic FX operators that can represent
arbitrary ONNX operators.

The operators are called "symbolic" because they don't do any actual computation
but instead serve as placeholders in the computation graph.

Each implementation contains two parts: A "real" implementation that produce all
zeros based on the input shape and dtype, and a "fake" implementation that does more
or less the same thing but is required by the `torch.library.custom_op` interface.
"""
_INT_TYPE = ...
_FLOAT_TYPE = ...
_STRING_TYPE = ...
_INT_SEQ_TYPE = ...
_FLOAT_SEQ_TYPE = ...
_STRING_SEQ_TYPE = ...

@dataclasses.dataclass
class EncodedAttrs:
    attr_keys: list[str]
    attr_types: list[str]
    attr_pos: list[tuple[int, int]]
    attr_ints: list[int]
    attr_floats: list[float]
    attr_strs: list[str]
    @classmethod
    def from_dict(
        cls,
        attrs: dict[
            str,
            int
            | float
            | str
            | bool
            | Sequence[int]
            | Sequence[float]
            | Sequence[str]
            | Sequence[bool],
        ],
    ) -> EncodedAttrs: ...
    def to_dict(
        self,
    ) -> dict[str, int | float | str | list[int] | list[float] | list[str]]: ...

@_symbolic.register_fake
def _(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtype: int,
    *,
    shape: Sequence[int | torch.SymInt],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = ...,
    metadata_props_values: Sequence[str] = ...,
    domain: str = ...,
    version: int | None = ...,
) -> torch.Tensor: ...
@_symbolic_multi_out.register_fake
def _(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtypes: Sequence[int],
    *,
    shapes: Sequence[Sequence[int | torch.SymInt]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = ...,
    metadata_props_values: Sequence[str] = ...,
    domain: str = ...,
    version: int | None = ...,
) -> list[torch.Tensor]: ...
