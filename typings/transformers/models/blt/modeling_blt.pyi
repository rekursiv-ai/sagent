from torch import nn

import torch

from .configuration_blt import (
    BltConfig,
    BltGlobalTransformerConfig,
    BltLocalDecoderConfig,
    BltLocalEncoderConfig,
    BltPatcherConfig,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

class BltMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class BltRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class BltRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: BltConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class BltTransformerLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = ...,
        cross_attention_mask: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor] | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...
def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class BltSelfAttention(nn.Module):
    def __init__(self, config: BltConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        use_cache: bool = ...,
        past_key_values=...,
        cache_position=...,
        **kwargs,
    ):  # -> tuple[Any, Any]:
        ...

class BltCrossAttention(nn.Module):
    def __init__(
        self, config: BltConfig, layer_idx: int, hidden_size: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

@auto_docstring
class BltPreTrainedModel(PreTrainedModel):
    config: BltConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _can_compile_fullgraph = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

class BltLocalEncoder(BltPreTrainedModel):
    config: BltLocalEncoderConfig
    _can_record_outputs = ...
    def __init__(self, config: BltLocalEncoderConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        patch_embeds: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        num_patches: int | None = ...,
        patch_ids: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> tuple[Tensor | Any, Tensor | None]:
        ...
    def patch_reduce(self, hidden_states, max_num_patches, patch_ids):  # -> Tensor:
        ...

class BltLocalDecoder(BltPreTrainedModel):
    config: BltLocalDecoderConfig
    def __init__(self, config: BltLocalDecoderConfig) -> None: ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        patch_embeds: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> Any:
        ...

class BltGlobalTransformer(BltPreTrainedModel):
    config: BltGlobalTransformerConfig
    _can_record_outputs = ...
    def __init__(self, config: BltGlobalTransformerConfig) -> None: ...
    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> Tensor | Any:
        ...

def process_patch_lengths(
    patch_lengths: torch.Tensor, max_patch_length: int | None
) -> torch.Tensor: ...

class BltPatcher(BltPreTrainedModel):
    config: BltPatcherConfig
    def __init__(self, config: BltPatcherConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        patch_size: int | None = ...,
        threshold: float | None = ...,
        max_patch_length: int | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> tuple[Tensor, Tensor, Any]:
        ...
    @staticmethod
    def patch_lengths_from_entropies(
        entropies, sequence_length, patch_size=..., threshold=...
    ):  # -> Tensor:
        ...

def rolling_polynomial_hash(token_tensor, prime: int = ...):  # -> Tensor:
    ...
def byte_group_hash_function(
    token_ids: torch.Tensor,
    group_size: int = ...,
    prime: int = ...,
    max_hash: int = ...,
):  # -> Tensor:
    ...
def compute_hash_embeddings(
    local_encoder_tokens: torch.Tensor,
    local_encoder,
    encoder_hash_tok_embedding: nn.Embedding,
    encoder_hash_byte_group_nb_functions: int,
    encoder_hash_byte_group_size: list,
    encoder_hash_byte_group_vocab: int,
) -> torch.Tensor: ...

class BltModel(BltPreTrainedModel):
    def __init__(self, config: BltConfig) -> None: ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        patch_lengths: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...

@auto_docstring(custom_intro=...)
class BltForCausalLM(BltPreTrainedModel, GenerationMixin):
    config: BltConfig
    _can_compile_fullgraph = ...
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config: BltConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cross_attention_states: torch.LongTensor | None = ...,
        cross_attention_mask: torch.LongTensor | None = ...,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor] | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast: ...

__all__ = ["BltForCausalLM", "BltModel", "BltPatcher", "BltPreTrainedModel"]
