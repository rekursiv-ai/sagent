from collections.abc import Callable
from typing import Any

from torch import nn

import torch

from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2MLP,
    Gemma2PreTrainedModel,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
)
from ...cache_utils import Cache, EncoderDecoderCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

_CHECKPOINT_FOR_DOC = ...
logger = ...

class T5GemmaModuleConfig(Gemma2Config): ...

class T5GemmaConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        encoder: T5GemmaModuleConfig | dict[Any, Any] | None = ...,
        decoder: T5GemmaModuleConfig | dict[Any, Any] | None = ...,
        is_encoder_decoder: bool = ...,
        dropout_rate: float = ...,
        classifier_dropout_rate: float = ...,
        attention_dropout: float = ...,
        tie_word_embeddings: bool = ...,
        vocab_size: int = ...,
        **kwargs,
    ) -> None: ...
    def __setattr__(self, key, value):  # -> None:
        ...

class T5GemmaRMSNorm(Gemma2RMSNorm): ...

class T5GemmaMLP(Gemma2MLP):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class T5GemmaRotaryEmbedding(Gemma2RotaryEmbedding):
    def __init__(self, config, device=...) -> None: ...

class T5GemmaSelfAttention(Gemma2Attention):
    def __init__(self, config: T5GemmaModuleConfig, layer_idx: int) -> None: ...

class T5GemmaCrossAttention(Gemma2Attention):
    def __init__(self, config: T5GemmaModuleConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

def bidirectional_mask_function(attention_mask: torch.Tensor | None) -> Callable: ...
def sliding_window_bidirectional_mask_function(sliding_window: int) -> Callable: ...

class T5GemmaEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.FloatTensor,]: ...

class T5GemmaDecoderLayer(T5GemmaEncoderLayer):
    def __init__(self, config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> torch.FloatTensor: ...

class T5GemmaClassificationHead(nn.Module):
    def __init__(
        self, hidden_size: int, num_labels: int, classifier_dropout_rate: float = ...
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class T5GemmaLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class T5GemmaPreTrainedModel(Gemma2PreTrainedModel):
    config: T5GemmaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

def make_default_2d_attention_mask(
    token_ids: torch.LongTensor | None,
    hidden_states: torch.Tensor,
    pad_token_id: int | None,
) -> torch.Tensor: ...

class T5GemmaEncoder(T5GemmaPreTrainedModel):
    _can_record_outputs = ...
    def __init__(self, config) -> None: ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

class T5GemmaDecoder(T5GemmaEncoder):
    _can_record_outputs = ...
    def __init__(self, config) -> None: ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring
class T5GemmaModel(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig) -> None: ...
    def get_encoder(self):  # -> T5GemmaEncoder:
        ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        decoder_position_ids: torch.LongTensor | None = ...,
        encoder_outputs: BaseModelOutput | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput: ...

@auto_docstring
class T5GemmaEncoderModel(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

class T5GemmaForConditionalGeneration(T5GemmaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def __init__(self, config: T5GemmaConfig) -> None: ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def get_encoder(self):  # -> T5GemmaEncoder:
        ...
    def get_decoder(self):  # -> T5GemmaDecoder:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        decoder_position_ids: torch.LongTensor | None = ...,
        encoder_outputs: BaseModelOutput | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | Seq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

@auto_docstring
class T5GemmaForSequenceClassification(T5GemmaPreTrainedModel):
    def __init__(
        self, config: T5GemmaConfig, is_encoder_decoder: bool | None = ...
    ) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        decoder_position_ids: torch.LongTensor | None = ...,
        encoder_outputs: BaseModelOutput | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput: ...

@auto_docstring
class T5GemmaForTokenClassification(T5GemmaPreTrainedModel):
    def __init__(
        self, config: T5GemmaConfig, is_encoder_decoder: bool | None = ...
    ) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        decoder_position_ids: torch.LongTensor | None = ...,
        encoder_outputs: BaseModelOutput | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput: ...

__all__ = [
    "T5GemmaConfig",
    "T5GemmaEncoderModel",
    "T5GemmaForConditionalGeneration",
    "T5GemmaForSequenceClassification",
    "T5GemmaForTokenClassification",
    "T5GemmaModel",
    "T5GemmaModuleConfig",
    "T5GemmaPreTrainedModel",
]
