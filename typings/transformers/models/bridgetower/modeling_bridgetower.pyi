from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_bridgetower import (
    BridgeTowerConfig,
    BridgeTowerTextConfig,
    BridgeTowerVisionConfig,
)
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch BridgeTower Model"""
logger = ...
_TOKENIZER_FOR_DOC = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`BridgeTowerModel`].
    """
)
class BridgeTowerModelOutput(ModelOutput):
    text_features: torch.FloatTensor | None = ...
    image_features: torch.FloatTensor | None = ...
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class BridgeTowerContrastiveOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    text_embeds: tuple[torch.FloatTensor] | None = ...
    image_embeds: tuple[torch.FloatTensor] | None = ...
    cross_embeds: tuple[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class BridgeTowerResidualAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def attention(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ):  # -> Any:
        ...
    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = ...
    ):  # -> Tensor:
        ...

class BridgeTowerTransformer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = ...
    ):  # -> list[Any]:
        ...

class BridgeTowerVisionEmbeddings(nn.Module):
    def __init__(self, config: BridgeTowerVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BridgeTowerVisionTransformer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask,
        interpolate_pos_encoding: bool = ...,
    ):  # -> Tensor:
        ...
    def forward_pre(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...
    ):  # -> Any:
        ...
    def forward_post(self, hidden_state: torch.Tensor):  # -> Any:
        ...

class BridgeTowerLinkTower(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, cross_modal_hidden_states, attention_mask
    ):  # -> Any:
        ...

class BridgeTowerSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BridgeTowerIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BridgeTowerOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BridgeTowerPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BridgeTowerSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=..., layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...

BRIDGE_TOWER_SELF_ATTENTION_CLASSES = ...

class BridgeTowerAttention(nn.Module):
    def __init__(self, config, position_embedding_type=..., layer_idx=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...

class BridgeTowerBertCrossLayer(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_attention_mask=...,
        past_key_values=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class BridgeTowerTextLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class BridgeTowerTextEncoder(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...

class BridgeTowerTextEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids=...,
        token_type_ids=...,
        position_ids=...,
        inputs_embeds=...,
        past_key_values_length=...,
    ):  # -> Any:
        ...
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):  # -> Tensor:
        ...

def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...

@auto_docstring
class BridgeTowerPreTrainedModel(PreTrainedModel):
    config: BridgeTowerConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...

class BridgeTowerVisionModel(BridgeTowerPreTrainedModel):
    config: BridgeTowerVisionConfig
    def __init__(self, config) -> None: ...
    @property
    def dtype(self):  # -> dtype:
        ...
    def forward(self, image, image_mask=..., interpolate_pos_encoding=...):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class BridgeTowerTextModel(BridgeTowerPreTrainedModel):
    config: BridgeTowerTextConfig
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class BridgeTowerModel(BridgeTowerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        image_token_type_idx: int | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple[torch.Tensor] | BridgeTowerModelOutput: ...
    def get_cls_features(self, text_features, image_features):  # -> Tensor:
        ...

class BridgeTowerPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class BridgeTowerMLMHead(nn.Module):
    def __init__(self, config, weight=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class BridgeTowerITMHead(nn.Module):
    def __init__(self, hidden_size) -> None: ...
    def forward(self, x):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class BridgeTowerForMaskedLM(BridgeTowerPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.LongTensor | None = ...,
    ) -> MaskedLMOutput | tuple[torch.FloatTensor]: ...

@auto_docstring(custom_intro=...)
class BridgeTowerForImageAndTextRetrieval(BridgeTowerPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.LongTensor | None = ...,
    ) -> SequenceClassifierOutput | tuple[torch.FloatTensor]: ...

class BridgeTowerContrastiveHead(nn.Module):
    def __init__(self, hidden_size, embed_size) -> None: ...
    def forward(self, x):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class BridgeTowerForContrastiveLearning(BridgeTowerPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        return_loss: bool | None = ...,
    ) -> BridgeTowerContrastiveOutput | tuple[torch.FloatTensor]: ...

__all__ = [
    "BridgeTowerForContrastiveLearning",
    "BridgeTowerForImageAndTextRetrieval",
    "BridgeTowerForMaskedLM",
    "BridgeTowerModel",
    "BridgeTowerPreTrainedModel",
]
