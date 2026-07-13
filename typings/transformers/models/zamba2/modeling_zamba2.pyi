from typing import Any

from torch import nn

import torch

from .configuration_zamba2 import Zamba2Config
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available

if is_mamba_ssm_available(): ...
else: ...
if is_causal_conv1d_available(): ...
else: ...
logger = ...

class Zamba2RMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, group_size, eps=...) -> None: ...
    def forward(self, hidden_states, gate=...): ...

class Zamba2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class Zamba2HybridDynamicCache:
    is_compileable = ...
    def __init__(
        self,
        config: Zamba2Config,
        batch_size: int,
        dtype: torch.dtype = ...,
        device: str | None = ...,
    ) -> None: ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
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
    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_position: torch.LongTensor,
    ) -> torch.Tensor: ...
    def reset(self):  # -> None:
        ...

class Zamba2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Zamba2Config, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...
def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class Zamba2Attention(nn.Module):
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

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):  # -> Tensor:
    ...
def reshape_into_chunks(input_tensor, pad_size, chunk_size):  # -> Tensor:
    ...
def segment_sum(input_tensor):  # -> Tensor:
    ...

is_fast_path_available = ...

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

class Zamba2AttentionDecoderLayer(nn.Module):
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

class Zamba2MambaDecoderLayer(nn.Module):
    def __init__(self, config: Zamba2Config, layer_idx: int) -> None: ...
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
        cache_position: torch.LongTensor | None = ...,
        transformer_hidden_states: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class Zamba2HybridLayer(nn.Module):
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

@auto_docstring
class Zamba2Model(Zamba2PreTrainedModel):
    def __init__(self, config: Zamba2Config) -> None: ...
    @auto_docstring
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
    def get_layers(self, blocks, linear_layers, mamba_layers):  # -> list[Any]:
        ...

class Zamba2ForCausalLM(Zamba2PreTrainedModel, GenerationMixin):
    def __init__(self, config: Zamba2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Zamba2HybridDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...

@auto_docstring(custom_intro=...)
class Zamba2ForSequenceClassification(Zamba2PreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutputWithPast: ...

__all__ = [
    "Zamba2ForCausalLM",
    "Zamba2ForSequenceClassification",
    "Zamba2Model",
    "Zamba2PreTrainedModel",
]
