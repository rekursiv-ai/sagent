from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch SeamlessM4Tv2 model."""
logger = ...
SEAMLESS_M4T_V2_COMMON_CUSTOM_ARGS = ...

@dataclass
@auto_docstring(custom_intro=...)
class SeamlessM4Tv2GenerationOutput(ModelOutput):
    waveform: torch.FloatTensor | None = ...
    waveform_lengths: torch.IntTensor | None = ...
    sequences: tuple[torch.FloatTensor] | None = ...
    unit_sequences: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class SeamlessM4Tv2TextToUnitDecoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    padding_mask: torch.Tensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class SeamlessM4Tv2TextToUnitOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    padding_mask: torch.Tensor | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    loss: torch.FloatTensor | None = ...

def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...
def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...
def format_speech_generation_kwargs(
    kwargs,
):  # -> tuple[dict[Any, Any], dict[Any, Any]]:
    ...

class SeamlessM4Tv2ConformerFeatureProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SeamlessM4Tv2ConformerFeedForward(nn.Module):
    def __init__(self, config, act_fn=..., dropout=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SeamlessM4Tv2ConformerConvolutionModule(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, attention_mask=...):  # -> Any:
        ...

class SeamlessM4Tv2ConformerSelfAttention(nn.Module):
    def __init__(self, config, use_position_embeddings=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class SeamlessM4Tv2ConformerEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        conv_attention_mask: torch.Tensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...

class SeamlessM4Tv2ConformerEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class SeamlessM4Tv2ConformerAdapterLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ):  # -> Any:
        ...

class SeamlessM4Tv2ConformerAdapter(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, attention_mask):  # -> Any:
        ...

class SeamlessM4Tv2ScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float | None = ...,
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
        ...

class SeamlessM4Tv2SinusoidalPositionalEmbedding(nn.Module):
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = ...
    ) -> None: ...
    def make_weights(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = ...
    ):  # -> None:
        ...
    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: int | None = ...
    ):  # -> Tensor:
        ...
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        past_key_values_length: int = ...,
    ):  # -> Tensor | Any:
        ...
    def create_position_ids_from_inputs_embeds(
        self, inputs_embeds, past_key_values_length
    ): ...

class SeamlessM4Tv2Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: SeamlessM4Tv2Config | None = ...,
        layer_idx: int | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class SeamlessM4Tv2FeedForwardNetwork(nn.Module):
    def __init__(self, config: SeamlessM4Tv2Config, ffn_dim: int) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SeamlessM4Tv2EncoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = ...,
    ) -> torch.Tensor: ...

class SeamlessM4Tv2DecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        layer_idx=...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

class SeamlessM4Tv2TextToUnitDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        padding_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> torch.Tensor: ...

@auto_docstring
class SeamlessM4Tv2PreTrainedModel(PreTrainedModel):
    config: SeamlessM4Tv2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2SpeechEncoder(SeamlessM4Tv2PreTrainedModel):
    main_input_name = ...
    def __init__(self, config: SeamlessM4Tv2Config) -> None: ...
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | Wav2Vec2BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2Encoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: nn.Embedding | None = ...,
        is_t2u_encoder: bool = ...,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2Decoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self, config: SeamlessM4Tv2Config, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2TextToUnitDecoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self, config: SeamlessM4Tv2Config, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        char_input_ids: torch.LongTensor | None = ...,
        char_count_per_id: torch.LongTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SeamlessM4Tv2TextToUnitDecoderOutput: ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2TextToUnitModel(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens_decoder: nn.Embedding | None = ...,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        char_input_ids: torch.LongTensor | None = ...,
        char_count_per_id: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2TextToUnitForConditionalGeneration(
    SeamlessM4Tv2PreTrainedModel, GenerationMixin
):
    _keys_to_ignore_on_load_missing = ...
    _tied_weights_keys = ...
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens_decoder: nn.Embedding | None = ...,
    ) -> None: ...
    def get_encoder(self):  # -> SeamlessM4Tv2Encoder:
        ...
    def get_decoder(self):  # -> SeamlessM4Tv2TextToUnitDecoder:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        char_input_ids: torch.LongTensor | None = ...,
        char_count_per_id: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> Seq2SeqLMOutput | tuple[torch.FloatTensor]: ...

class HifiGanResidualBlock(nn.Module):
    def __init__(
        self, channels, kernel_size=..., dilation=..., leaky_relu_slope=...
    ) -> None: ...
    def get_padding(self, kernel_size, dilation=...): ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...
    def forward(self, hidden_states): ...

class SeamlessM4Tv2VariancePredictor(nn.Module):
    def __init__(
        self, embed_dim, hidden_dim, kernel_size, var_pred_dropout
    ) -> None: ...
    def forward(
        self, hidden_states: Tensor, padding_mask: Tensor | None = ...
    ) -> Tensor: ...

class SeamlessM4Tv2HifiGan(nn.Module):
    def __init__(self, config: SeamlessM4Tv2Config) -> None: ...
    def forward(self, input_embeds: torch.FloatTensor) -> torch.FloatTensor: ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2CodeHifiGan(PreTrainedModel):
    config: SeamlessM4Tv2Config
    main_input_name = ...
    _no_split_modules = ...
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor,
        speaker_id: torch.Tensor,
        lang_id: torch.Tensor,
    ) -> tuple[torch.Tensor]: ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2ForTextToText(SeamlessM4Tv2PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = ...
    main_input_name = ...
    _tied_weights_keys = ...
    def __init__(self, config: SeamlessM4Tv2Config) -> None: ...
    def get_encoder(self):  # -> SeamlessM4Tv2Encoder:
        ...
    def get_decoder(self):  # -> SeamlessM4Tv2Decoder:
        ...
    def get_input_embeddings(self):  # -> SeamlessM4Tv2ScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring(custom_args=SEAMLESS_M4T_V2_COMMON_CUSTOM_ARGS)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> Seq2SeqLMOutput | tuple[torch.FloatTensor]: ...
    def generate(
        self,
        input_ids=...,
        tgt_lang=...,
        generation_config=...,
        logits_processor=...,
        stopping_criteria=...,
        prefix_allowed_tokens_fn=...,
        synced_gpus=...,
        **kwargs,
    ):  # -> GenerateOutput | LongTensor:
        ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2ForSpeechToText(SeamlessM4Tv2PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = ...
    main_input_name = ...
    _tied_weights_keys = ...
    def __init__(self, config: SeamlessM4Tv2Config) -> None: ...
    def get_encoder(self):  # -> SeamlessM4Tv2SpeechEncoder:
        ...
    def get_decoder(self):  # -> SeamlessM4Tv2Decoder:
        ...
    def get_input_embeddings(self):  # -> SeamlessM4Tv2ScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring(custom_args=SEAMLESS_M4T_V2_COMMON_CUSTOM_ARGS)
    def forward(
        self,
        input_features: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> Seq2SeqLMOutput | tuple[torch.FloatTensor]: ...
    def generate(
        self,
        input_features=...,
        tgt_lang=...,
        generation_config=...,
        logits_processor=...,
        stopping_criteria=...,
        prefix_allowed_tokens_fn=...,
        synced_gpus=...,
        **kwargs,
    ):  # -> GenerateOutput | LongTensor:
        ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2ForTextToSpeech(SeamlessM4Tv2PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = ...
    main_input_name = ...
    _tied_weights_keys = ...
    def __init__(self, config: SeamlessM4Tv2Config) -> None: ...
    def get_encoder(self):  # -> SeamlessM4Tv2Encoder:
        ...
    def get_decoder(self):  # -> SeamlessM4Tv2Decoder:
        ...
    def get_input_embeddings(self):  # -> SeamlessM4Tv2ScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring(custom_args=SEAMLESS_M4T_V2_COMMON_CUSTOM_ARGS)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> Seq2SeqLMOutput | tuple[torch.FloatTensor]: ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = ...,
        return_intermediate_token_ids: bool | None = ...,
        tgt_lang: str | None = ...,
        speaker_id: int | None = ...,
        **kwargs,
    ) -> torch.Tensor | SeamlessM4Tv2GenerationOutput: ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2ForSpeechToSpeech(SeamlessM4Tv2PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = ...
    main_input_name = ...
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_encoder(self):  # -> SeamlessM4Tv2SpeechEncoder:
        ...
    def get_decoder(self):  # -> SeamlessM4Tv2Decoder:
        ...
    def get_input_embeddings(self):  # -> SeamlessM4Tv2ScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring(custom_args=SEAMLESS_M4T_V2_COMMON_CUSTOM_ARGS)
    def forward(
        self,
        input_features: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> Seq2SeqLMOutput | tuple[torch.FloatTensor]: ...
    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor | None = ...,
        return_intermediate_token_ids: bool | None = ...,
        tgt_lang: str | None = ...,
        speaker_id: int | None = ...,
        **kwargs,
    ) -> torch.Tensor | SeamlessM4Tv2GenerationOutput: ...

@auto_docstring(custom_intro=...)
class SeamlessM4Tv2Model(SeamlessM4Tv2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config, current_modality=...) -> None: ...
    def set_modality(self, modality=...):  # -> None:
        ...
    def get_encoder(self):  # -> SeamlessM4Tv2Encoder | SeamlessM4Tv2SpeechEncoder:
        ...
    def get_input_embeddings(self):  # -> SeamlessM4Tv2ScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring(custom_args=SEAMLESS_M4T_V2_COMMON_CUSTOM_ARGS)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> Seq2SeqLMOutput | tuple[torch.FloatTensor]: ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = ...,
        input_features: torch.Tensor | None = ...,
        return_intermediate_token_ids: bool | None = ...,
        tgt_lang: str | None = ...,
        speaker_id: int | None = ...,
        generate_speech: bool | None = ...,
        **kwargs,
    ) -> torch.Tensor | SeamlessM4Tv2GenerationOutput: ...

__all__ = [
    "SeamlessM4Tv2ForSpeechToSpeech",
    "SeamlessM4Tv2ForSpeechToText",
    "SeamlessM4Tv2ForTextToSpeech",
    "SeamlessM4Tv2ForTextToText",
    "SeamlessM4Tv2Model",
    "SeamlessM4Tv2PreTrainedModel",
]
