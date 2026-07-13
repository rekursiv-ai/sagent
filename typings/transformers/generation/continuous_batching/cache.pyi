import torch

from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import GenerationConfig
from ...utils.metrics import attach_tracer, traced

def group_layers_by_attn_type(
    config: PretrainedConfig,
) -> tuple[list[list[int]], list[str]]: ...

@attach_tracer()
class PagedAttentionCache:
    def __init__(
        self,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        device: torch.device,
        dtype: torch.dtype = ...,
        layer_device_map: dict[int, str | torch.device | int] | None = ...,
        tp_size: int | None = ...,
    ) -> None: ...
    @traced
    def allocate_blocks(self, n_blocks: int, request_id: str) -> int: ...
    @traced
    def free_blocks(self, request_id: str) -> None: ...
    def get_num_free_blocks(self) -> int: ...
    @traced
    def extend_read_indices(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        read_index: list[list[int]],
    ) -> None: ...
    @traced
    def extend_write_indices(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        write_index: list[list[int]],
    ) -> None: ...
    @traced
    def get_seqlens_k(
        self, request_id: str, past_length: int, query_length: int
    ) -> dict[str, int]: ...
    @traced
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        read_index: list[torch.Tensor],
        write_index: list[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class PagedAttentionMemoryHandler:
    _activation_dtype = ...
    _input_dtype = ...
    _upper_bound_max_batch_tokens = ...
    _upper_bound_num_blocks = ...
    def __init__(
        self,
        block_size: int,
        page_size: int,
        num_groups: int,
        group_size: int,
        peak_activation_per_token: int,
        num_attention_masks: int,
    ) -> None: ...
    @staticmethod
    def get_available_memory(max_memory_percent: float = ...) -> int: ...
    def infer_num_blocks_and_max_batch_tokens(
        self,
        num_blocks: int | None = ...,
        max_batch_tokens: int | None = ...,
        max_memory_percent: float = ...,
        cache_dtype: torch.dtype = ...,
    ) -> tuple[int, int]: ...
    def compute_num_blocks_and_max_batch_tokens(
        self,
        max_memory_percent: float = ...,
        cache_dtype: torch.dtype = ...,
        m: float = ...,
    ) -> tuple[int, int]: ...
    def compute_max_batch_tokens(
        self,
        num_blocks: int,
        max_memory_percent: float = ...,
        cache_dtype: torch.dtype = ...,
    ) -> int: ...
    def compute_num_blocks(
        self,
        max_batch_tokens: int,
        max_memory_percent: float = ...,
        cache_dtype: torch.dtype = ...,
    ) -> int: ...
    def compute_memory_footprint(
        self,
        num_blocks: int | None = ...,
        max_batch_tokens: int | None = ...,
        cache_dtype: torch.dtype = ...,
    ) -> tuple[int, int, int]: ...
