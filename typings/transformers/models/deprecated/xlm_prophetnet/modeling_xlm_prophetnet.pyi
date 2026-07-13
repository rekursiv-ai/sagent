from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_xlm_prophetnet import XLMProphetNetConfig
from ....cache_utils import Cache
from ....modeling_layers import GradientCheckpointingLayer
from ....modeling_outputs import BaseModelOutput
from ....modeling_utils import PreTrainedModel
from ....utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ....utils.deprecation import deprecate_kwarg

"""PyTorch XLM-ProphetNet model."""
logger = ...
_CONFIG_FOR_DOC = ...
XLM_PROPHETNET_START_DOCSTRING = ...
XLM_PROPHETNET_INPUTS_DOCSTRING = ...
XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING = ...

def softmax(hidden_state, dim, onnx_trace=...):  # -> Tensor:
    ...
def ngram_attention_bias(sequence_length, ngram, device, dtype):  # -> Tensor:
    ...
def compute_relative_buckets(
    num_buckets, max_distance, relative_positions, is_bidirectional=...
):  # -> Tensor:
    ...
def compute_all_stream_relative_buckets(
    num_buckets, max_distance, position_ids
):  # -> tuple[Any | Tensor, Any | Tensor]:
    ...

@dataclass
class XLMProphetNetSeq2SeqLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    logits_ngram: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_ngram_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    decoder_ngram_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    @property
    def decoder_cross_attentions(self):  # -> tuple[FloatTensor] | None:
        ...

@dataclass
class XLMProphetNetSeq2SeqModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_ngram_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    decoder_ngram_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    @property
    def decoder_cross_attentions(self):  # -> tuple[FloatTensor] | None:
        ...

@dataclass
class XLMProphetNetDecoderModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    hidden_states_ngram: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    ngram_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
class XLMProphetNetDecoderLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    logits_ngram: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    hidden_states_ngram: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    ngram_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

class XLMProphetNetPreTrainedModel(PreTrainedModel):
    config: XLMProphetNetConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

class XLMProphetNetPositionalEmbeddings(nn.Embedding):
    def __init__(self, config: XLMProphetNetConfig) -> None: ...
    def forward(
        self,
        inputs_shape,
        device,
        attention_mask=...,
        past_key_values=...,
        position_ids=...,
    ):  # -> tuple[Tensor, Tensor | Any]:
        ...

class XLMProphetNetAttention(nn.Module):
    def __init__(self, config: XLMProphetNetConfig, num_attn_heads: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        key_value_states: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        layer_head_mask: Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[Tensor, Tensor | None]: ...

class XLMProphetNetFeedForward(nn.Module):
    def __init__(self, config: XLMProphetNetConfig, ffn_dim: int) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class XLMProphetNetNgramSelfAttention(nn.Module):
    def __init__(self, config: XLMProphetNetConfig) -> None: ...
    def prepare_for_onnx_export_(self):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        past_key_values: Cache | None = ...,
        attention_mask=...,
        layer_head_mask=...,
        extended_predict_attention_mask=...,
        main_relative_position_buckets=...,
        predict_relative_position_buckets=...,
        position_ids=...,
    ):  # -> tuple[Tensor, Tensor, Tensor, Cache | None]:
        ...
    def get_main_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, main_relative_position_buckets
    ):  # -> Tensor:
        ...
    def get_predict_relative_pos_embeddings(
        self,
        hidden_states,
        attn_weights,
        position_ids,
        predict_relative_position_buckets,
    ):  # -> Tensor:
        ...

class XLMProphetNetEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: XLMProphetNetConfig) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class XLMProphetNetDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: XLMProphetNetConfig) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attn_mask=...,
        layer_head_mask=...,
        cross_attn_layer_head_mask=...,
        extended_predict_attention_mask=...,
        main_relative_position_buckets=...,
        predict_relative_position_buckets=...,
        position_ids=...,
        past_key_values=...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any | None, ...] | tuple[Any, ...] | tuple[Any, Any, Any, Any | None] | tuple[Any]:
        ...

@add_start_docstrings(
    ...,
    XLM_PROPHETNET_START_DOCSTRING,
)
class XLMProphetNetEncoder(XLMProphetNetPreTrainedModel):
    def __init__(
        self, config: XLMProphetNetConfig, word_embeddings: nn.Embedding | None = ...
    ) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@add_start_docstrings(
    ...,
    XLM_PROPHETNET_START_DOCSTRING,
)
class XLMProphetNetDecoder(XLMProphetNetPreTrainedModel):
    def __init__(
        self, config: XLMProphetNetConfig, word_embeddings: nn.Embedding | None = ...
    ) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=XLMProphetNetDecoderModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | XLMProphetNetDecoderModelOutput: ...
    def compute_buffered_relative_buckets(
        self, position_ids
    ):  # -> tuple[Tensor | Any, Tensor]:
        ...
    def prepare_attention_mask(self, hidden_states, attention_mask): ...
    def prepare_predict_attention_mask(self, hidden_states, attention_mask): ...

@add_start_docstrings(
    ...,
    XLM_PROPHETNET_START_DOCSTRING,
)
class XLMProphetNetModel(XLMProphetNetPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: XLMProphetNetConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> XLMProphetNetEncoder:
        ...
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=XLMProphetNetSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.Tensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | XLMProphetNetSeq2SeqModelOutput: ...

@add_start_docstrings(
    ...,
    XLM_PROPHETNET_START_DOCSTRING,
)
class XLMProphetNetForConditionalGeneration(XLMProphetNetPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: XLMProphetNetConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=XLMProphetNetSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.Tensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | XLMProphetNetSeq2SeqLMOutput: ...
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=...,
        attention_mask=...,
        head_mask=...,
        decoder_head_mask=...,
        cross_attn_head_mask=...,
        use_cache=...,
        encoder_outputs=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...
    def get_encoder(self):  # -> XLMProphetNetEncoder:
        ...
    def get_decoder(self):  # -> XLMProphetNetDecoder:
        ...

@add_start_docstrings(
    ...,
    XLM_PROPHETNET_START_DOCSTRING,
)
class XLMProphetNetForCausalLM(XLMProphetNetPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: XLMProphetNetConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> XLMProphetNetDecoder:
        ...
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=XLMProphetNetDecoderLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | XLMProphetNetDecoderLMOutput: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        head_mask=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...

class XLMProphetNetDecoderWrapper(XLMProphetNetPreTrainedModel):
    def __init__(self, config: XLMProphetNetConfig) -> None: ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

__all__ = [
    "XLMProphetNetDecoder",
    "XLMProphetNetEncoder",
    "XLMProphetNetForCausalLM",
    "XLMProphetNetForConditionalGeneration",
    "XLMProphetNetModel",
    "XLMProphetNetPreTrainedModel",
]
