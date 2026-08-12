from collections.abc import Callable
from typing import Any

from torch import nn

import torch

from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2ForCausalLM,
    Gemma2MLP,
    Gemma2Model,
    Gemma2PreTrainedModel,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
)
from ..paligemma.modeling_paligemma import (
    PaligemmaCausalLMOutputWithPast,
    PaliGemmaForConditionalGeneration,
    PaliGemmaModel,
    PaligemmaModelOutputWithPast,
)
from ..siglip import SiglipVisionConfig
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForSequenceClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

logger = ...

class Gemma3TextConfig(Gemma2Config, PretrainedConfig):
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        head_dim=...,
        hidden_activation=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        query_pre_attn_scalar=...,
        sliding_window=...,
        layer_types=...,
        final_logit_softcapping=...,
        attn_logit_softcapping=...,
        rope_scaling=...,
        rope_local_base_freq=...,
        use_bidirectional_attention=...,
        **kwargs,
    ) -> None: ...

class Gemma3Config(PretrainedConfig):
    def __init__(
        self,
        text_config: Gemma3TextConfig | dict[str, Any] | None = ...,
        vision_config: SiglipVisionConfig | dict[str, Any] | None = ...,
        mm_tokens_per_image: int = ...,
        boi_token_index: int = ...,
        eoi_token_index: int = ...,
        image_token_index: int = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

class Gemma3ModelOutputWithPast(PaligemmaModelOutputWithPast): ...
class Gemma3CausalLMOutputWithPast(PaligemmaCausalLMOutputWithPast): ...

class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = ...,
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
        ...

class Gemma3MLP(Gemma2MLP):
    def __init__(self, config: Gemma3TextConfig) -> None: ...

class Gemma3RMSNorm(Gemma2RMSNorm):
    def __init__(self, dim: int, eps: float = ...) -> None: ...

class Gemma3RotaryEmbedding(Gemma2RotaryEmbedding):
    def __init__(self, config: Gemma3TextConfig, device=...) -> None: ...

class Gemma3Attention(Gemma2Attention):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

GEMMA3_START_DOCSTRING = ...

class Gemma3PreTrainedModel(Gemma2PreTrainedModel):
    base_model_prefix = ...
    _no_split_modules = ...

class Gemma3TextModel(Gemma2Model):
    config: Gemma3TextConfig
    def __init__(self, config: Gemma3TextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPast: ...

class Gemma3ForCausalLM(Gemma2ForCausalLM):
    config: Gemma3TextConfig
    base_model_prefix = ...
    def __init__(self, config: Gemma3TextConfig) -> None: ...

class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config) -> None: ...
    def forward(self, vision_outputs: torch.Tensor):  # -> Tensor:
        ...

def token_type_ids_mask_function(
    token_type_ids: torch.Tensor | None,
    image_group_ids: torch.Tensor | None,
    tokens_per_image: int,
) -> Callable | None: ...

class Gemma3Model(PaliGemmaModel):
    accepts_loss_kwargs = ...
    def __init__(self, config: Gemma3Config) -> None: ...
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **lm_kwargs,
    ) -> tuple | Gemma3ModelOutputWithPast: ...

class Gemma3ForConditionalGeneration(PaliGemmaForConditionalGeneration):
    accepts_loss_kwargs = ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **lm_kwargs,
    ) -> tuple | Gemma3CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        pixel_values=...,
        attention_mask=...,
        token_type_ids=...,
        use_cache=...,
        logits_to_keep=...,
        labels=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    @staticmethod
    def create_masks_for_generate(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        token_type_ids: torch.Tensor | None = ...,
        **kwargs,
    ) -> dict: ...

class Gemma3ForSequenceClassification(Gemma3PreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast: ...

class Gemma3TextForSequenceClassification(
    GenericForSequenceClassification, Gemma3PreTrainedModel
):
    config: Gemma3TextConfig

__all__ = [
    "Gemma3Config",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3ForSequenceClassification",
    "Gemma3Model",
    "Gemma3PreTrainedModel",
    "Gemma3TextConfig",
    "Gemma3TextForSequenceClassification",
    "Gemma3TextModel",
]
