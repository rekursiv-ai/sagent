from torch import nn

import torch

from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSpectrogramOutput,
)
from ...modeling_utils import EmbeddingAccessMixin, PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch SpeechT5 model."""
logger = ...
_HIDDEN_STATES_START_POSITION = ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...
def shift_spectrograms_right(
    input_values: torch.Tensor,
    reduction_factor: int = ...,
    attention_mask: torch.Tensor | None = ...,
):  # -> tuple[Tensor, Tensor | None]:
    ...

class SpeechT5NoLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class SpeechT5LayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class SpeechT5GroupNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class SpeechT5SinusoidalPositionalEmbedding(nn.Module):
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
        self, input_ids: torch.Tensor, past_key_values_length: int = ...
    ):  # -> Tensor | Any:
        ...
    def create_position_ids_from_input_ids(
        self,
        input_ids: torch.Tensor,
        padding_idx: int,
        past_key_values_length: int | None = ...,
    ):  # -> Tensor:
        ...

class SpeechT5PositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class SpeechT5ScaledPositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=...) -> None: ...
    def forward(self, emb):  # -> Any:
        ...

class SpeechT5RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_length=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SpeechT5SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None: ...
    def forward(self, hidden_states): ...

class SpeechT5FeatureEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_values):  # -> Any:
        ...

class SpeechT5FeatureProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class SpeechT5SpeechEncoderPrenet(nn.Module):
    def __init__(self, config) -> None: ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.LongTensor | None = ...,
        mask_time_indices: torch.FloatTensor | None = ...,
    ):  # -> tuple[Any, LongTensor | None]:
        ...

class SpeechT5SpeechDecoderPrenet(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_values: torch.Tensor,
        speaker_embeddings: torch.Tensor | None = ...,
    ):  # -> Tensor | Any:
        ...

class SpeechT5BatchNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SpeechT5SpeechDecoderPostnet(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> tuple[Any, Tensor | Any, Any]:
        ...
    def postnet(self, hidden_states: torch.Tensor):  # -> Tensor | Any:
        ...

class SpeechT5TextEncoderPrenet(nn.Module, EmbeddingAccessMixin):
    def __init__(self, config) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Any:
        ...

class SpeechT5TextDecoderPrenet(nn.Module, EmbeddingAccessMixin):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
    ):  # -> tuple[Any, LongTensor | None]:
        ...

class SpeechT5TextDecoderPostnet(nn.Module, EmbeddingAccessMixin):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Any:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...

class SpeechT5Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float | None = ...,
        is_decoder: bool | None = ...,
        bias: bool | None = ...,
        layer_idx: bool | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        position_bias: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...

class SpeechT5FeedForward(nn.Module):
    def __init__(self, config, intermediate_size) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SpeechT5EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        position_bias: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class SpeechT5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SpeechT5Config, layer_idx=...) -> None: ...
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
    ):  # -> tuple[Tensor, Any, Any | None] | tuple[Tensor]:
        ...

@auto_docstring
class SpeechT5PreTrainedModel(PreTrainedModel):
    config: SpeechT5Config
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

class SpeechT5Encoder(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class SpeechT5EncoderWithSpeechPrenet(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class SpeechT5EncoderWithTextPrenet(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class SpeechT5EncoderWithoutPrenet(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class SpeechT5Decoder(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

class SpeechT5DecoderWithSpeechPrenet(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def forward(
        self,
        input_values: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        speaker_embeddings: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

class SpeechT5DecoderWithTextPrenet(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        input_values: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

class SpeechT5DecoderWithoutPrenet(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def forward(
        self,
        input_values: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

class SpeechT5GuidedMultiheadAttentionLoss(nn.Module):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def forward(
        self,
        attentions: torch.FloatTensor,
        input_masks: torch.BoolTensor,
        output_masks: torch.BoolTensor,
    ) -> torch.Tensor: ...

class SpeechT5SpectrogramLoss(nn.Module):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def forward(
        self,
        attention_mask: torch.LongTensor,
        outputs_before_postnet: torch.FloatTensor,
        outputs_after_postnet: torch.FloatTensor,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        cross_attentions: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class SpeechT5Model(SpeechT5PreTrainedModel):
    def __init__(
        self,
        config: SpeechT5Config,
        encoder: nn.Module | None = ...,
        decoder: nn.Module | None = ...,
    ) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> SpeechT5EncoderWithoutPrenet | Module:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_values: torch.Tensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        decoder_head_mask: torch.FloatTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        speaker_embeddings: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class SpeechT5ForSpeechToText(SpeechT5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: SpeechT5Config) -> None: ...
    def get_encoder(self):  # -> SpeechT5EncoderWithoutPrenet | Module:
        ...
    def get_decoder(self):  # -> PreTrainedModel | SpeechT5Model:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        decoder_head_mask: torch.FloatTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | Seq2SeqLMOutput: ...

@auto_docstring(custom_intro=...)
class SpeechT5ForTextToSpeech(SpeechT5PreTrainedModel):
    main_input_name = ...
    def __init__(self, config: SpeechT5Config) -> None: ...
    @classmethod
    def can_generate(cls) -> bool: ...
    def get_encoder(self):  # -> SpeechT5EncoderWithoutPrenet | Module:
        ...
    def get_decoder(self):  # -> PreTrainedModel | SpeechT5Model:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_values: torch.FloatTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        decoder_head_mask: torch.FloatTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        speaker_embeddings: torch.FloatTensor | None = ...,
        labels: torch.FloatTensor | None = ...,
        stop_labels: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | Seq2SeqSpectrogramOutput: ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = ...,
        speaker_embeddings: torch.FloatTensor | None = ...,
        threshold: float = ...,
        minlenratio: float = ...,
        maxlenratio: float = ...,
        vocoder: nn.Module | None = ...,
        output_cross_attentions: bool = ...,
        return_output_lengths: bool = ...,
        **kwargs,
    ) -> torch.FloatTensor | tuple[torch.FloatTensor, torch.FloatTensor]: ...
    @torch.no_grad()
    def generate_speech(
        self,
        input_ids: torch.LongTensor,
        speaker_embeddings: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        threshold: float = ...,
        minlenratio: float = ...,
        maxlenratio: float = ...,
        vocoder: nn.Module | None = ...,
        output_cross_attentions: bool = ...,
        return_output_lengths: bool = ...,
    ) -> torch.FloatTensor | tuple[torch.FloatTensor, torch.FloatTensor]: ...

@auto_docstring(custom_intro=...)
class SpeechT5ForSpeechToSpeech(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config) -> None: ...
    def get_encoder(self):  # -> SpeechT5EncoderWithoutPrenet | Module:
        ...
    def get_decoder(self):  # -> PreTrainedModel | SpeechT5Model:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_values: torch.FloatTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        decoder_head_mask: torch.FloatTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        speaker_embeddings: torch.FloatTensor | None = ...,
        labels: torch.FloatTensor | None = ...,
        stop_labels: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | Seq2SeqSpectrogramOutput: ...
    @torch.no_grad()
    def generate_speech(
        self,
        input_values: torch.FloatTensor,
        speaker_embeddings: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        threshold: float = ...,
        minlenratio: float = ...,
        maxlenratio: float = ...,
        vocoder: nn.Module | None = ...,
        output_cross_attentions: bool = ...,
        return_output_lengths: bool = ...,
    ) -> torch.FloatTensor: ...

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

@auto_docstring(
    custom_intro="""
    HiFi-GAN vocoder.
    """
)
class SpeechT5HifiGan(PreTrainedModel):
    config: SpeechT5HifiGanConfig
    main_input_name = ...
    def __init__(self, config: SpeechT5HifiGanConfig) -> None: ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...
    @auto_docstring(custom_intro=...)
    def forward(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor: ...

__all__ = [
    "SpeechT5ForSpeechToSpeech",
    "SpeechT5ForSpeechToText",
    "SpeechT5ForTextToSpeech",
    "SpeechT5HifiGan",
    "SpeechT5Model",
    "SpeechT5PreTrainedModel",
]
