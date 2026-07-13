from dataclasses import dataclass

from torch import nn

import torch

from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig
from ...cache_utils import Cache
from ...generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from ...generation.streamers import BaseStreamer
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    ModelOutput,
    Seq2SeqLMOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Musicgen model."""
logger = ...

@dataclass
@auto_docstring
class MusicgenUnconditionalInput(ModelOutput):
    encoder_outputs: tuple[torch.FloatTensor] | None = ...
    attention_mask: torch.LongTensor | None = ...
    guidance_scale: float | None = ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

class MusicgenSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int) -> None: ...
    def make_weights(self, num_embeddings: int, embedding_dim: int):  # -> None:
        ...
    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):  # -> Tensor:
        ...
    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, past_key_values_length: int = ...
    ):  # -> Tensor | Any:
        ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = ...,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class MusicgenAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float | None = ...,
        is_decoder: bool | None = ...,
        bias: bool | None = ...,
        is_causal: bool | None = ...,
        config: MusicgenConfig | None = ...,
        layer_idx: int | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class MusicgenDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MusicgenDecoderConfig, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        cross_attn_layer_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

@auto_docstring
class MusicgenPreTrainedModel(PreTrainedModel):
    config: MusicgenDecoderConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

class MusicgenDecoder(MusicgenPreTrainedModel):
    def __init__(self, config: MusicgenDecoderConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring
class MusicgenModel(MusicgenPreTrainedModel):
    def __init__(self, config: MusicgenDecoderConfig) -> None: ...
    def get_input_embeddings(self):  # -> ModuleList:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class MusicgenForCausalLM(MusicgenPreTrainedModel, GenerationMixin):
    def __init__(self, config: MusicgenDecoderConfig) -> None: ...
    def get_input_embeddings(self):  # -> ModuleList:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> ModuleList:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> MusicgenDecoder:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        head_mask=...,
        cross_attn_head_mask=...,
        past_key_values=...,
        use_cache=...,
        delay_pattern_mask=...,
        guidance_scale=...,
        **kwargs,
    ):  # -> dict[str, Tensor | Any | bool | None]:
        ...
    def build_delay_pattern_mask(
        self,
        input_ids: torch.LongTensor,
        pad_token_id: int,
        max_length: int | None = ...,
    ):  # -> tuple[Tensor, Tensor] | tuple[LongTensor, Tensor]:
        ...
    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):  # -> Tensor:
        ...
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = ...,
        generation_config: GenerationConfig | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
        stopping_criteria: StoppingCriteriaList | None = ...,
        synced_gpus: bool | None = ...,
        streamer: BaseStreamer | None = ...,
        **kwargs,
    ):  # -> GenerateNonBeamOutput | LongTensor | Tensor:
        ...

@auto_docstring(custom_intro=...)
class MusicgenForConditionalGeneration(MusicgenPreTrainedModel, GenerationMixin):
    config: MusicgenConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    def __init__(
        self,
        config: MusicgenConfig | None = ...,
        text_encoder: PreTrainedModel | None = ...,
        audio_encoder: PreTrainedModel | None = ...,
        decoder: MusicgenForCausalLM | None = ...,
    ) -> None: ...
    def tie_weights(self):  # -> None:
        ...
    def get_audio_encoder(self):  # -> PreTrainedModel | None:
        ...
    def get_text_encoder(self):  # -> PreTrainedModel | None:
        ...
    def get_encoder(self):  # -> PreTrainedModel | None:
        ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def get_output_embeddings(self):  # -> ModuleList:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @classmethod
    def from_sub_models_pretrained(
        cls,
        text_encoder_pretrained_model_name_or_path: str | None = ...,
        audio_encoder_pretrained_model_name_or_path: str | None = ...,
        decoder_pretrained_model_name_or_path: str | None = ...,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.BoolTensor | None = ...,
        input_values: torch.FloatTensor | None = ...,
        padding_mask: torch.BoolTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        encoder_outputs: tuple[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | Seq2SeqLMOutput: ...
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=...,
        attention_mask=...,
        head_mask=...,
        decoder_attention_mask=...,
        decoder_head_mask=...,
        cross_attn_head_mask=...,
        use_cache=...,
        encoder_outputs=...,
        decoder_delay_pattern_mask=...,
        guidance_scale=...,
        cache_position=...,
        **kwargs,
    ):  # -> dict[str, Any | Tensor | None]:
        ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...
    def resize_token_embeddings(self, *args, **kwargs): ...
    def freeze_audio_encoder(self):  # -> None:
        ...
    def freeze_text_encoder(self):  # -> None:
        ...
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = ...,
        generation_config: GenerationConfig | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
        stopping_criteria: StoppingCriteriaList | None = ...,
        synced_gpus: bool | None = ...,
        streamer: BaseStreamer | None = ...,
        **kwargs,
    ):  # -> GenerateNonBeamOutput | LongTensor | Any | Tensor:
        ...
    def get_unconditional_inputs(
        self, num_samples=...
    ):  # -> MusicgenUnconditionalInput:
        ...

__all__ = [
    "MusicgenForCausalLM",
    "MusicgenForConditionalGeneration",
    "MusicgenModel",
    "MusicgenPreTrainedModel",
]
