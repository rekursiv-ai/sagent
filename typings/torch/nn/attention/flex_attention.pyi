from collections.abc import Callable
from enum import Enum
from typing import NamedTuple, NotRequired, TypedDict

from torch import Tensor
from torch._prims_common import DeviceLikeType

import torch

"""This module implements the user facing API for flex_attention in PyTorch."""
_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = ...
_WARNINGS_SHOWN: set[str] = ...
__all__ = [
    "AuxOutput",
    "AuxRequest",
    "BlockMask",
    "FlexKernelOptions",
    "and_masks",
    "create_block_mask",
    "create_mask",
    "create_nested_block_mask",
    "flex_attention",
    "noop_mask",
    "or_masks",
]
type _score_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
type _mask_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]

class FlexKernelOptions(TypedDict, total=False):
    num_warps: NotRequired[int]
    num_stages: NotRequired[int]
    BLOCK_M: NotRequired[int]
    BLOCK_N: NotRequired[int]
    BLOCK_M1: NotRequired[int]
    BLOCK_N1: NotRequired[int]
    BLOCK_M2: NotRequired[int]
    BLOCK_N2: NotRequired[int]
    PRESCALE_QK: NotRequired[bool]
    ROWS_GUARANTEED_SAFE: NotRequired[bool]
    BLOCKS_ARE_CONTIGUOUS: NotRequired[bool]
    WRITE_DQ: NotRequired[bool]
    FORCE_USE_FLEX_ATTENTION: NotRequired[bool]
    USE_TMA: NotRequired[bool]
    kpack: NotRequired[int]
    matrix_instr_nonkdim: NotRequired[int]
    waves_per_eu: NotRequired[int]

class AuxRequest(NamedTuple):
    lse: bool = ...
    max_scores: bool = ...

class AuxOutput(NamedTuple):
    lse: Tensor | None = ...
    max_scores: Tensor | None = ...

class _ModificationType(Enum):
    SCORE_MOD = ...
    MASK_MOD = ...
    UNKNOWN = ...

def noop_mask(
    batch: Tensor, head: Tensor, token_q: Tensor, token_kv: Tensor
) -> Tensor: ...

_DEFAULT_SPARSE_BLOCK_SIZE = ...
_LARGE_SPARSE_BLOCK_SIZE = ...

class BlockMask:
    seq_lengths: tuple[int, int]
    kv_num_blocks: Tensor
    kv_indices: Tensor
    full_kv_num_blocks: Tensor | None
    full_kv_indices: Tensor | None
    q_num_blocks: Tensor | None
    q_indices: Tensor | None
    full_q_num_blocks: Tensor | None
    full_q_indices: Tensor | None
    BLOCK_SIZE: tuple[int, int]
    mask_mod: _mask_mod_signature
    def __init__(
        self,
        seq_lengths: tuple[int, int],
        kv_num_blocks: Tensor,
        kv_indices: Tensor,
        full_kv_num_blocks: Tensor | None,
        full_kv_indices: Tensor | None,
        q_num_blocks: Tensor | None,
        q_indices: Tensor | None,
        full_q_num_blocks: Tensor | None,
        full_q_indices: Tensor | None,
        BLOCK_SIZE: tuple[int, int],
        mask_mod: _mask_mod_signature,
    ) -> None: ...
    @classmethod
    def from_kv_blocks(
        cls,
        kv_num_blocks: Tensor,
        kv_indices: Tensor,
        full_kv_num_blocks: Tensor | None = ...,
        full_kv_indices: Tensor | None = ...,
        BLOCK_SIZE: int | tuple[int, int] = ...,
        mask_mod: _mask_mod_signature | None = ...,
        seq_lengths: tuple[int, int] | None = ...,
        compute_q_blocks: bool = ...,
    ) -> Self: ...
    def as_tuple(
        self, flatten: bool = ...
    ) -> tuple[int | tuple[int, int] | Tensor | _mask_mod_signature | None, ...]: ...
    @property
    def shape(self) -> tuple[*tuple[int, ...], int, int]: ...
    def __getitem__(self, index) -> BlockMask: ...
    def numel(self) -> Any: ...
    def sparsity(self) -> float: ...
    def to_dense(self) -> Tensor: ...
    def to_string(self, grid_size=..., limit=...) -> LiteralString: ...
    def to(self, device: torch.device | str) -> BlockMask: ...

def or_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature: ...
def and_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature: ...
def create_mask(
    mod_fn: _score_mod_signature | _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: DeviceLikeType = ...,
) -> Tensor: ...
def create_block_mask(
    mask_mod: _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: DeviceLikeType = ...,
    BLOCK_SIZE: int | tuple[int, int] = ...,
    _compile=...,
) -> BlockMask: ...
def create_nested_block_mask(
    mask_mod: _mask_mod_signature,
    B: int | None,
    H: int | None,
    q_nt: torch.Tensor,
    kv_nt: torch.Tensor | None = ...,
    BLOCK_SIZE: int | tuple[int, int] = ...,
    _compile=...,
) -> BlockMask: ...
def flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: _score_mod_signature | None = ...,
    block_mask: BlockMask | None = ...,
    scale: float | None = ...,
    enable_gqa: bool = ...,
    return_lse: bool = ...,
    kernel_options: FlexKernelOptions | None = ...,
    *,
    return_aux: AuxRequest | None = ...,
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, AuxOutput]: ...
