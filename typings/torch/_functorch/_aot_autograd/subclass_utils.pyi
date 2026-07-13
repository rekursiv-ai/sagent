from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from torch import SymInt, Tensor
from torch.types import IntLikeType

import torch

from .descriptors import AOTInput, AOTOutput
from .schemas import (
    FxValue,
    MemoryFormatMeta,
    PlainTensorMeta,
    SubclassCreationMeta,
    ViewAndMutationMeta,
)

"""
This file contains utilities for tracing through __torch_dispatch__ based tensor subclasses and modes.
AOTAutograd's responsibility is to trace through all pytorch capabilities that live in the pytorch dispatcher,
and this includes tensor subclasses that implement __torch_dispatch__.
"""
zip = ...
T = TypeVar("T", bound=torch.Tensor)

def requires_subclass_dispatch(args, fw_metadata: ViewAndMutationMeta) -> bool: ...
def maybe_suggest_memory_format(
    t, with_memory_format: bool
) -> MemoryFormatMeta | None: ...
def get_subclass_typing_container(
    tensor_subclass: torch.Tensor,
) -> dict[type[torch.Tensor], list[type[torch.Tensor]]]: ...
def create_subclass_metadata(
    a: Any, start_idx: int, count_symints: bool, with_memory_format: bool = ...
) -> tuple[PlainTensorMeta, int] | tuple[SubclassCreationMeta, int | Any]: ...
def create_subclass_meta(
    curr_args: list[Any] | tuple[Any, ...],
    *,
    count_symints: bool = ...,
    with_memory_format: bool = ...,
) -> list[PlainTensorMeta | SubclassCreationMeta]: ...
def enumerate_filter_symints(
    lst: Iterable[IntLikeType],
) -> list[tuple[int, SymInt]]: ...
def compute_symint_placeholders(lst: Iterable[None | int | SymInt]) -> list[bool]: ...

AOTDescriptor = TypeVar("AOTDescriptor", AOTInput, AOTOutput)

def unwrap_tensor_subclasses(
    wrapped_args: list[FxValue],
    wrapped_args_descs: list[AOTDescriptor],
    *,
    append_symints: bool,
) -> tuple[list[FxValue], list[AOTDescriptor]]: ...
def runtime_unwrap_tensor_subclasses(
    wrapped_args: list[Tensor | int],
    *,
    append_symints: bool,
    subclass_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = ...,
) -> list[int | Tensor | SymInt]: ...
def unwrap_tensor_subclasses_with_indices_to_original(
    wrapped_args,
) -> tuple[list[Any], list[Any]]: ...
def remap_unwrapped_subclass_arg_indices(
    wrapped_args, static_input_indices
) -> list[Any]: ...
def wrap_tensor_subclasses(
    unwrapped_args: tuple[Any, ...] | list[Any],
    *,
    subclass_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None = ...,
    included_subclass_symints: bool = ...,
    is_runtime: bool = ...,
    make_subclass_override: Callable | None = ...,
) -> tuple[Any, ...]: ...
def wrap_tensor_subclasses_maybe_joint(
    unwrapped_args, *, is_joint_structure: bool, meta: ViewAndMutationMeta
) -> tuple[Any, ...] | list[Any]: ...
def compute_inner_mutated_inp_indices_from_subclass_meta(
    fw_metadata: ViewAndMutationMeta, inner_metadata: ViewAndMutationMeta
) -> list[int]: ...
