from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_kosmos2_5 import (
    Kosmos2_5Config,
    Kosmos2_5TextConfig,
    Kosmos2_5VisionConfig,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    replace_return_docstrings,
)

"""PyTorch KOSMOS-2.5 model."""
logger = ...
_CONFIG_FOR_DOC = Kosmos2_5Config

def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...

KOSMOS2_5_START_DOCSTRING = ...
KOSMOS2_5_VISION_INPUTS_DOCSTRING = ...
KOSMOS2_5_TEXT_INPUTS_DOCSTRING = ...
KOSMOS2_5_INPUTS_DOCSTRING = ...

@dataclass
class Kosmos2_5ModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    width: torch.FloatTensor | None = ...
    height: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    projection_attentions: tuple[torch.FloatTensor] | None = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

@dataclass
class Kosmos2_5ForConditionalGenerationModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    width: torch.FloatTensor | None = ...
    height: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    projection_attentions: tuple[torch.FloatTensor] | None = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class Kosmos2_5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

class Kosmos2_5VisionEmbeddings(nn.Module):
    def __init__(self, config: Kosmos2_5VisionConfig) -> None: ...
    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor: ...

class Kosmos2_5VisionMlp(nn.Module):
    def __init__(self, config: Kosmos2_5VisionConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class Kosmos2_5VisionAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, attention_mask=..., **kwargs: Unpack[TransformersKwargs]
    ):  # -> tuple[Any, Any]:
        ...

class Kosmos2_5VisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: Kosmos2_5VisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class Kosmos2_5VisionEncoder(nn.Module):
    def __init__(self, config: Kosmos2_5VisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

class Kosmos2_5TextSinusoidalPositionalEmbedding(nn.Module):
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
        position_ids: torch.Tensor | None = ...,
    ):  # -> Tensor | Any:
        ...
    def create_position_ids_from_inputs_embeds(
        self, inputs_embeds, past_key_values_length
    ): ...

class Kosmos2_5TextFFN(nn.Module):
    def __init__(self, config: Kosmos2_5TextConfig) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class Kosmos2_5TextAttention(nn.Module):
    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal=...,
        layer_idx: int | None = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Kosmos2_5TextBlock(GradientCheckpointingLayer):
    def __init__(self, config: Kosmos2_5TextConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class Kosmos2_5TextTransformer(nn.Module):
    def __init__(self, config: Kosmos2_5TextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        image_embeds: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions: ...

class Kosmos2_5ImageToTextProjection(nn.Module):
    def __init__(self, config: Kosmos2_5Config) -> None: ...
    def forward(self, features):  # -> tuple[Any, Any]:
        ...

class Kosmos2_5PreTrainedModel(PreTrainedModel):
    config_class = Kosmos2_5Config
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn_2 = ...
    _supports_cache_class = ...
    _supports_sdpa = ...
    _supports_attention_backend = ...

class Kosmos2_5VisionModel(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5VisionConfig
    def __init__(self, config: Kosmos2_5VisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> Linear:
        ...
    def forward(
        self,
        flattened_patches: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

class Kosmos2_5TextModel(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5TextConfig
    def __init__(self, config: Kosmos2_5TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(KOSMOS2_5_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=Kosmos2_5TextConfig,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        image_embeds: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions: ...

@add_start_docstrings(
    ...,
    KOSMOS2_5_START_DOCSTRING,
)
class Kosmos2_5Model(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5Config
    def __init__(self, config: Kosmos2_5Config) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @add_start_docstrings_to_model_forward(KOSMOS2_5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Kosmos2_5ModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        flattened_patches: torch.Tensor | None = ...,
        width: torch.Tensor | None = ...,
        height: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        image_embeds: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Kosmos2_5ModelOutput: ...

@add_start_docstrings(
    ...,
    KOSMOS2_5_START_DOCSTRING,
)
class Kosmos2_5TextForCausalLM(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5TextConfig
    _tied_weights_keys = ...
    def __init__(self, config: Kosmos2_5TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(KOSMOS2_5_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithCrossAttentions, config_class=Kosmos2_5TextConfig
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        image_embeds: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithCrossAttentions: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        image_embeds=...,
        image_embeds_position_mask=...,
        past_key_values=...,
        attention_mask=...,
        use_cache=...,
        cache_position=...,
        position_ids=...,
        **model_kwargs,
    ):  # -> dict[str, Any | Tensor | None]:
        ...

@add_start_docstrings(
    ...,
    KOSMOS2_5_START_DOCSTRING,
)
class Kosmos2_5ForConditionalGeneration(Kosmos2_5PreTrainedModel, GenerationMixin):
    config_class = Kosmos2_5Config
    _tied_weights_keys = ...
    def __init__(self, config: Kosmos2_5Config) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @can_return_tuple
    @add_start_docstrings_to_model_forward(KOSMOS2_5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Kosmos2_5ForConditionalGenerationModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        flattened_patches: torch.Tensor | None = ...,
        width: torch.Tensor | None = ...,
        height: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        image_embeds: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Kosmos2_5ForConditionalGenerationModelOutput: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches=...,
        image_embeds=...,
        image_embeds_position_mask=...,
        past_key_values=...,
        attention_mask=...,
        use_cache=...,
        cache_position=...,
        position_ids=...,
        **model_kwargs,
    ):  # -> dict[str, Any | Tensor | None]:
        ...

__all__ = [
    "Kosmos2_5ForConditionalGeneration",
    "Kosmos2_5Model",
    "Kosmos2_5PreTrainedModel",
]
