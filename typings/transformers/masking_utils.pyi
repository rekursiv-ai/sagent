from collections.abc import Callable

from torch.nn.attention.flex_attention import BlockMask

import torch

from .cache_utils import Cache
from .configuration_utils import PretrainedConfig
from .utils.generic import GeneralInterface
from .utils.import_utils import is_torch_flex_attn_available

if is_torch_flex_attn_available(): ...
else:
    BlockMask = ...
_is_torch_greater_or_equal_than_2_5 = ...
_is_torch_greater_or_equal_than_2_6 = ...
_is_torch_xpu_available = ...
logger = ...

def and_masks(*mask_functions: Callable) -> Callable: ...
def or_masks(*mask_functions: Callable) -> Callable: ...
def causal_mask_function(
    batch_idx: int, head_idx: int, q_idx: int, kv_idx: int
) -> bool: ...
def sliding_window_overlay(sliding_window: int) -> Callable: ...
def chunked_overlay(chunk_size: int, left_padding: torch.Tensor) -> Callable: ...
def sliding_window_causal_mask_function(sliding_window: int) -> Callable: ...
def chunked_causal_mask_function(
    chunk_size: int, left_padding: torch.Tensor
) -> Callable: ...
def padding_mask_function(padding_mask: torch.Tensor) -> Callable: ...
def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable: ...
def add_offsets_to_mask_function(
    mask_function: Callable, q_offset: int, kv_offset: int
) -> Callable: ...
def prepare_padding_mask(
    attention_mask: torch.Tensor | None,
    kv_length: int,
    kv_offset: int,
    _slice: bool = ...,
) -> torch.Tensor | None: ...
def sdpa_mask_recent_torch(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = ...,
    mask_function: Callable = ...,
    attention_mask: torch.Tensor | None = ...,
    local_size: int | None = ...,
    allow_is_causal_skip: bool = ...,
    **kwargs,
) -> torch.Tensor | None: ...
def sdpa_mask_older_torch(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = ...,
    mask_function: Callable = ...,
    attention_mask: torch.Tensor | None = ...,
    local_size: int | None = ...,
    allow_is_causal_skip: bool = ...,
    allow_torch_fix: bool = ...,
    **kwargs,
) -> torch.Tensor | None: ...

sdpa_mask = ...

def eager_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = ...,
    mask_function: Callable = ...,
    attention_mask: torch.Tensor | None = ...,
    dtype: torch.dtype = ...,
    **kwargs,
) -> torch.Tensor: ...
def flash_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = ...,
    mask_function: Callable = ...,
    attention_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> Tensor | None:
    ...
def flex_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = ...,
    mask_function: Callable = ...,
    attention_mask: torch.Tensor | None = ...,
    **kwargs,
) -> BlockMask: ...

class AttentionMaskInterface(GeneralInterface):
    _global_mapping = ...

ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = ...

def find_packed_sequence_indices(position_ids: torch.Tensor) -> torch.Tensor: ...
def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = ...,
    or_mask_function: Callable | None = ...,
    and_mask_function: Callable | None = ...,
) -> torch.Tensor | BlockMask | None: ...
def create_sliding_window_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = ...,
    or_mask_function: Callable | None = ...,
    and_mask_function: Callable | None = ...,
) -> torch.Tensor | BlockMask | None: ...
def create_chunked_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = ...,
    or_mask_function: Callable | None = ...,
    and_mask_function: Callable | None = ...,
) -> torch.Tensor | BlockMask | None: ...

LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = ...

def create_masks_for_generate(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = ...,
    or_mask_function: Callable | None = ...,
    and_mask_function: Callable | None = ...,
    **kwargs,
):  # -> dict[Any, Any] | Tensor | None:
    ...

GREEN = ...
YELLOW = ...
RESET = ...
BLACK_SQUARE = ...
WHITE_SQUARE = ...
GREY_SQUARE = ...
LOW_TRIANGLE = ...
UPPER_TRIANGLE = ...

def get_style(
    style,
):  # -> tuple[Literal['🀙', '█'], Literal['🀆', '░'], Literal['🀛', '▙'], Literal['🀛', '▜']]:
    ...

YELLOW_SQUARE = ...
GREEN_SQUARE = ...

def tensor_to_mask_visual(
    original_tensor: torch.Tensor, grid_size=..., style=...
) -> str: ...

class AttentionMask(torch.Tensor):
    def __new__(cls, data, style=...):  # -> Self:
        ...
    def __init__(self, data) -> None: ...
    def to_string(self, grid_size=..., limit=...):  # -> LiteralString:
        ...
    def __repr__(self):  # -> LiteralString:
        ...
    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, style: str | None = ...
    ) -> AttentionMask: ...
