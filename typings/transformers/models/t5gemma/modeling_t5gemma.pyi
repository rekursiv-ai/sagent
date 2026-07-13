from collections.abc import Callable

from torch import nn

import torch

from .configuration_t5gemma import T5GemmaConfig, T5GemmaModuleConfig
from ...cache_utils import Cache, EncoderDecoderCache
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
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

logger = ...

class T5GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = ...) -> None: ...
    def forward(self, x): ...
    def extra_repr(self):  # -> str:
        ...

class T5GemmaMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class T5GemmaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = ...,
    scaling: float | None = ...,
    softcap: float | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class T5GemmaSelfAttention(nn.Module):
    def __init__(self, config: T5GemmaModuleConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class T5GemmaCrossAttention(nn.Module):
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

class T5GemmaAttention(nn.Module):
    def __init__(self, config: T5GemmaConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

@auto_docstring
class T5GemmaPreTrainedModel(PreTrainedModel):
    config: T5GemmaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

def bidirectional_mask_function(attention_mask: torch.Tensor | None) -> Callable: ...
def sliding_window_bidirectional_mask_function(sliding_window: int) -> Callable: ...
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
    "T5GemmaEncoderModel",
    "T5GemmaForConditionalGeneration",
    "T5GemmaForSequenceClassification",
    "T5GemmaForTokenClassification",
    "T5GemmaModel",
    "T5GemmaPreTrainedModel",
]
