from torch import nn

import torch

from .configuration_speech_to_text_2 import Speech2Text2Config
from ....cache_utils import Cache
from ....modeling_layers import GradientCheckpointingLayer
from ....modeling_outputs import CausalLMOutputWithCrossAttentions
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, replace_return_docstrings
from ....utils.deprecation import deprecate_kwarg

"""PyTorch Speech2Text2 model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...

class Speech2Text2SinusoidalPositionalEmbedding(nn.Module):
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
    ):  # -> Tensor:
        ...
    def create_position_ids_from_input_ids(
        self,
        input_ids: torch.Tensor,
        padding_idx: int,
        past_key_values_length: int | None = ...,
    ):  # -> Tensor:
        ...

class Speech2Text2Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: Speech2Text2Config | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Speech2Text2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Speech2Text2Config) -> None: ...
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
    ):  # -> tuple[Tensor | Any | None, ...] | tuple[Tensor, ...] | tuple[Tensor, Any, Any | None] | tuple[Tensor]:
        ...

class Speech2Text2PreTrainedModel(PreTrainedModel):
    config: Speech2Text2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

SPEECH_TO_TEXT_2_START_DOCSTRING = ...

class Speech2Text2Decoder(Speech2Text2PreTrainedModel):
    def __init__(self, config: Speech2Text2Config) -> None: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        head_mask=...,
        cross_attn_head_mask=...,
        past_key_values=...,
        inputs_embeds=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ): ...

@add_start_docstrings(
    ...,
    SPEECH_TO_TEXT_2_START_DOCSTRING,
)
class Speech2Text2DecoderWrapper(Speech2Text2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

@add_start_docstrings(
    ...,
    SPEECH_TO_TEXT_2_START_DOCSTRING,
)
class Speech2Text2ForCausalLM(Speech2Text2PreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Speech2Text2Decoder:
        ...
    @replace_return_docstrings(
        output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | CausalLMOutputWithCrossAttentions: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...

__all__ = ["Speech2Text2ForCausalLM", "Speech2Text2PreTrainedModel"]
