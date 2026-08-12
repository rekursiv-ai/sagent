from typing import Any
from torch import nn

import torch

from .configuration_blt import (
    BltConfig,
    BltGlobalTransformerConfig,
    BltLocalDecoderConfig,
    BltLocalEncoderConfig,
    BltPatcherConfig,
)
from ..cohere2.modeling_cohere2 import Cohere2RotaryEmbedding
from ..mllama.modeling_mllama import (
    MllamaForCausalLM,
    MllamaPreTrainedModel,
    MllamaSelfAttentionDecoderLayer,
    MllamaTextCrossAttention,
    MllamaTextMLP,
    MllamaTextRMSNorm,
    MllamaTextSelfAttention,
)
from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs

"""Blt modular model, inheriting from Mllama where appropriate."""
logger = ...

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
def process_patch_lengths(
    patch_lengths: torch.Tensor, max_patch_length: int | None
) -> torch.Tensor: ...

class BltMLP(MllamaTextMLP): ...
class BltRMSNorm(MllamaTextRMSNorm): ...
class BltRotaryEmbedding(Cohere2RotaryEmbedding): ...

class BltTransformerLayer(MllamaSelfAttentionDecoderLayer):
    def __init__(self, config, layer_idx: int) -> None: ...

class BltSelfAttention(MllamaTextSelfAttention):
    def __init__(self, config: BltConfig, layer_idx: int) -> None: ...
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

class BltCrossAttention(MllamaTextCrossAttention):
    def __init__(
        self, config: BltConfig, layer_idx: int, hidden_size: int | None = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> tuple[Any, Any]:
        ...

@auto_docstring
class BltPreTrainedModel(MllamaPreTrainedModel):
    config: BltConfig
    _supports_attention_backend = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _no_split_modules = ...
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

class BltForCausalLM(MllamaForCausalLM):
    config: BltConfig
    _can_compile_fullgraph = ...
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config: BltConfig) -> None: ...
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
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | CausalLMOutputWithPast: ...

__all__ = ["BltForCausalLM", "BltModel", "BltPatcher", "BltPreTrainedModel"]
