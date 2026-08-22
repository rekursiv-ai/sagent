from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_ovis2 import Ovis2Config, Ovis2VisionConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple

@dataclass
@auto_docstring(custom_intro=...)
class Ovis2ModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Ovis2CausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: torch.FloatTensor | None = ...

@use_kernel_forward_from_hub("RMSNorm")
class Ovis2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class Ovis2VisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Ovis2VisionEmbeddings(nn.Module):
    def __init__(self, config: Ovis2VisionConfig) -> None: ...
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

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

class Ovis2VisionAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Ovis2MLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Ovis2Attention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Ovis2VisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Ovis2VisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Ovis2VisionEncoder(nn.Module):
    def __init__(self, config: Ovis2VisionConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

class Ovis2VisionTransformer(nn.Module):
    def __init__(self, config: Ovis2VisionConfig) -> None: ...
    @can_return_tuple
    def forward(
        self, pixel_values, attention_mask: torch.Tensor | None = ..., **kwargs
    ):  # -> BaseModelOutput:
        ...

class Ovis2VisualEmbeddingTable(nn.Embedding):
    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Ovis2PreTrainedModel(PreTrainedModel):
    config: Ovis2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_cache_class = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...

def hard_softmax(logits: torch.Tensor, dim: int):  # -> Tensor:
    ...

class Ovis2VisionModel(Ovis2PreTrainedModel):
    config: Ovis2VisionConfig
    def __init__(self, config: Ovis2VisionConfig) -> None: ...
    def forward(
        self, pixel_values: torch.FloatTensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

@auto_docstring(custom_intro=...)
class Ovis2Model(Ovis2PreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: Ovis2Config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(
        self, pixel_values: torch.FloatTensor
    ) -> torch.FloatTensor: ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Tensor | Any:
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
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | Ovis2ModelOutputWithPast: ...

@auto_docstring
class Ovis2ForConditionalGeneration(Ovis2PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: Ovis2Config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(self, pixel_values: torch.FloatTensor):  # -> FloatTensor:
        ...
    @property
    def language_model(self):  # -> Any:
        ...
    @property
    def vision_tower(self):  # -> Ovis2VisionModel:
        ...
    @property
    def multi_modal_projector(self): ...
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
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | Ovis2CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = ["Ovis2ForConditionalGeneration", "Ovis2Model", "Ovis2PreTrainedModel"]
