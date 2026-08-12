from typing import Any
from torch import nn

import torch

from .configuration_zamba2 import Zamba2Config
from ..llama.modeling_llama import LlamaRotaryEmbedding
from ..zamba.modeling_zamba import (
    ZambaAttention,
    ZambaAttentionDecoderLayer,
    ZambaForCausalLM,
    ZambaForSequenceClassification,
    ZambaHybridDynamicCache,
    ZambaHybridLayer,
    ZambaMambaDecoderLayer,
    ZambaModel,
    ZambaRMSNorm,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available

if is_mamba_ssm_available(): ...
else: ...
if is_causal_conv1d_available(): ...
else: ...
is_fast_path_available = ...
_CONFIG_FOR_DOC = ...
logger = ...

class Zamba2RMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, group_size, eps=...) -> None: ...
    def forward(self, hidden_states, gate=...): ...

class Zamba2RMSNorm(ZambaRMSNorm): ...

class Zamba2HybridDynamicCache(ZambaHybridDynamicCache):
    def __init__(
        self,
        config: Zamba2Config,
        batch_size: int,
        dtype: torch.dtype = ...,
        device: str | None = ...,
    ) -> None: ...
    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_position: torch.LongTensor,
    ) -> torch.Tensor: ...
    def reset(self):  # -> None:
        ...
    def get_seq_length(self, layer_idx: int | None = ...) -> int: ...

class Zamba2RotaryEmbedding(LlamaRotaryEmbedding): ...

class Zamba2Attention(ZambaAttention):
    def __init__(
        self,
        config: Zamba2Config,
        layer_idx: int | None = ...,
        num_fwd_mem_blocks: int | None = ...,
        block_id: int | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Zamba2HybridDynamicCache | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Zamba2MambaMixer(nn.Module):
    def __init__(self, config: Zamba2Config, layer_idx: int | None = ...) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Zamba2HybridDynamicCache | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    def torch_forward(
        self,
        input_states,
        cache_params: Zamba2HybridDynamicCache | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: Zamba2HybridDynamicCache | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...

class Zamba2MLP(nn.Module):
    def __init__(
        self,
        config: Zamba2Config,
        num_fwd_mem_blocks=...,
        block_id: int | None = ...,
    ) -> None: ...
    def forward(self, hidden_state, layer_idx=...):  # -> Any:
        ...

class Zamba2AttentionDecoderLayer(ZambaAttentionDecoderLayer):
    def __init__(
        self,
        config: Zamba2Config,
        block_id: int | None = ...,
        layer_idx: int | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Zamba2HybridDynamicCache | None = ...,
        output_attentions: bool | None = ...,
        position_embeddings: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class Zamba2MambaDecoderLayer(ZambaMambaDecoderLayer):
    def __init__(self, config: Zamba2Config, layer_idx: int) -> None: ...

class Zamba2HybridLayer(ZambaHybridLayer):
    def __init__(
        self,
        shared_transformer: Zamba2AttentionDecoderLayer,
        linear: nn.Linear,
        mamba: Zamba2MambaDecoderLayer,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor | None = ...,
        layer_idx: int | None = ...,
        attention_mask: torch.Tensor | None = ...,
        causal_mask: torch.Tensor | None = ...,
        past_key_values: Zamba2HybridDynamicCache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        position_embeddings: torch.LongTensor | None = ...,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class Zamba2PreTrainedModel(PreTrainedModel):
    config: Zamba2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_sdpa = ...
    _is_stateful = ...

class Zamba2Model(ZambaModel, Zamba2PreTrainedModel):
    def __init__(self, config: Zamba2Config) -> None: ...
    def get_layers(self, blocks, linear_layers, mamba_layers):  # -> list[Any]:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Zamba2HybridDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | BaseModelOutputWithPast: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | BaseModelOutputWithPast: ...

class Zamba2ForCausalLM(ZambaForCausalLM): ...
class Zamba2ForSequenceClassification(ZambaForSequenceClassification): ...

__all__ = [
    "Zamba2ForCausalLM",
    "Zamba2ForSequenceClassification",
    "Zamba2Model",
    "Zamba2PreTrainedModel",
]
