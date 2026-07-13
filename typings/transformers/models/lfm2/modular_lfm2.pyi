from typing import Any

from torch import nn

import torch

from .configuration_lfm2 import Lfm2Config
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import is_causal_conv1d_available

if is_causal_conv1d_available(): ...
else: ...
kernel_modules = ...
is_fast_path_available = ...
logger = ...

class Lfm2RMSNorm(LlamaRMSNorm): ...
class Lfm2RotaryEmbedding(LlamaRotaryEmbedding): ...

class Lfm2MLP(nn.Module):
    def __init__(self, config: Lfm2Config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Lfm2HybridConvCache:
    max_batch_size = ...
    is_compileable = ...
    key_cache = ...
    value_cache = ...
    def __init__(
        self,
        config: Lfm2Config,
        max_batch_size: int,
        dtype: torch.dtype = ...,
        device: torch.device | str | None = ...,
    ) -> None: ...
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def reorder_cache(self, beam_idx: torch.LongTensor):  # -> None:
        ...
    def get_seq_length(self, layer_idx: int | None = ...) -> int: ...
    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]: ...
    def crop(self, max_length: int):  # -> None:
        ...
    def __len__(self) -> int: ...
    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
    def reset(self):  # -> None:
        ...

class Lfm2Attention(LlamaAttention):
    def __init__(self, config: Lfm2Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Lfm2HybridConvCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Lfm2ShortConv(nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def cuda_kernels_forward(
        self,
        x: torch.Tensor,
        past_key_values: Lfm2HybridConvCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def slow_forward(
        self,
        x: torch.Tensor,
        past_key_values: Lfm2HybridConvCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Lfm2HybridConvCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...

class Lfm2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm2Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Lfm2HybridConvCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Lfm2PreTrainedModel(LlamaPreTrainedModel):
    _can_compile_fullgraph = ...

class Lfm2Model(LlamaModel):
    def __init__(self, config: Lfm2Config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Lfm2HybridConvCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...

class Lfm2ForCausalLM(LlamaForCausalLM): ...

__all__ = ["Lfm2ForCausalLM", "Lfm2Model", "Lfm2PreTrainedModel"]
