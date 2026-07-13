from dataclasses import dataclass

import queue
import threading

import torch

from .cache import PagedAttentionCache
from .requests import GenerationOutput
from .scheduler import Scheduler
from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import GenerationConfig
from ...utils.metrics import attach_tracer, traced

def build_attention_mask(
    attention_mask: torch.Tensor,
    cumulative_seqlens_q: torch.Tensor,
    cumulative_seqlens_k: torch.Tensor,
    sliding_window: int = ...,
) -> None: ...

@dataclass
class PagedAttentionArgs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor
    cumulative_seqlens_q: torch.Tensor
    cumulative_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    write_index: list[torch.Tensor]
    read_index: list[torch.Tensor]
    logits_indices: torch.Tensor
    cache: PagedAttentionCache
    use_cache: bool = ...

@attach_tracer()
class ContinuousBatchProcessor:
    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        model_device: torch.device,
        model_dtype: torch.dtype,
        scheduler: Scheduler,
        streaming: bool = ...,
        manual_eviction: bool = ...,
        slice_inputs: bool = ...,
    ) -> None: ...
    @traced(standalone=True)
    def setup_static_tensors(self, num_groups: int) -> None: ...
    def return_attention_mask(self) -> bool: ...
    @traced
    @torch.no_grad()
    def reset_static_tensors(self, full_reset: bool = ...):  # -> None:
        ...
    def get_model_kwargs(self) -> PagedAttentionArgs: ...
    def __repr__(self):  # -> str:
        ...
    @traced
    def prepare_next_batch(self) -> bool: ...
    @traced
    def update_batch(self):  # -> None:
        ...
    @traced
    def has_pending_requests(self) -> bool: ...
    @traced
    def handle_batch_error(self, error):  # -> None:
        ...
    @traced
    def fail_all_requests(self, error):  # -> None:
        ...

@attach_tracer()
class ContinuousBatchingManager:
    def __init__(
        self,
        model,
        generation_config: GenerationConfig,
        manual_eviction: bool = ...,
        max_queue_size=...,
        streaming: bool = ...,
        slice_inputs: bool = ...,
    ) -> None: ...
    @traced
    def start(self):  # -> None:
        ...
    def is_running(self):  # -> bool:
        ...
    def stop(self, block: bool = ..., timeout: float | None = ...):  # -> None:
        ...
    def join(self, timeout: float | None = ...):  # -> None:
        ...
    def add_request(
        self,
        input_ids: list[int],
        request_id: str | None = ...,
        max_new_tokens: int | None = ...,
    ) -> str: ...
    def add_requests(self, inputs: list[list[int]], **kwargs):  # -> None:
        ...
    def cancel_request(self, request_id: str):  # -> None:
        ...
    def get_result(self, request_id=..., timeout=...) -> GenerationOutput | None: ...
    def __iter__(self):  # -> Generator[GenerationOutput, Any, None]:
        ...
    def request_id_iter(self, request_id):  # -> Generator[GenerationOutput, Any, None]:
        ...
    @staticmethod
    def supported_attention_implementations() -> set[str]: ...
    @staticmethod
    def default_attention_implementation() -> str: ...
    @traced
    def warmup(self, batch_processor):  # -> None:
        ...
    @traced
    def evict_request_from_cache(self, request_id: str):  # -> None:
        ...

class ContinuousMixin:
    def init_continuous_batching(
        self,
        generation_config: GenerationConfig | None = ...,
        manual_eviction: bool = ...,
        max_queue_size: int = ...,
        streaming: bool = ...,
        slice_inputs: bool = ...,
    ) -> ContinuousBatchingManager: ...
    @traced
    @torch.inference_mode()
    def generate_batch(
        self,
        inputs: list[list[int]],
        generation_config: GenerationConfig | None = ...,
        progress_bar: bool = ...,
        slice_inputs: bool = ...,
        **kwargs,
    ) -> list[list[int]]: ...
