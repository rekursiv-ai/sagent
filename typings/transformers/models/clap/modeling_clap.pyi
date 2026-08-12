from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)

"""PyTorch CLAP model."""
logger = ...

def interpolate(hidden_states, ratio): ...
def window_partition(hidden_states, window_size): ...
def window_reverse(windows, window_size, height, width): ...
def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...

@dataclass
@auto_docstring(custom_intro=...)
class ClapTextModelOutput(ModelOutput):
    text_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class ClapAudioModelOutput(ModelOutput):
    audio_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring
class ClapOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_audio: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    audio_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    audio_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class ClapDropPath(nn.Module):
    def __init__(self, drop_prob=...) -> None: ...
    def forward(self, hidden_states): ...

class ClapAudioAFFBlock(nn.Module):
    def __init__(self, config: ClapAudioConfig) -> None: ...
    def forward(self, hidden_states, residual): ...

class ClapAudioPatchEmbed(nn.Module):
    def __init__(self, config: ClapAudioConfig) -> None: ...
    def forward(self, hidden_states, is_longer_idx=...):  # -> Any:
        ...

class ClapAudioSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class ClapAudioSelfOutput(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ClapAudioAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class ClapAudioIntermediate(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ClapAudioOutput(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ClapAudioLayer(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        num_heads,
        drop_path_rate=...,
        shift_size=...,
    ) -> None: ...
    def set_shift_and_window_size(self, input_resolution):  # -> None:
        ...
    def get_attn_mask(self, height, width, dtype, device):  # -> Tensor | None:
        ...
    def maybe_pad(
        self, hidden_states, height, width
    ):  # -> tuple[Tensor, tuple[Literal[0], Literal[0], Literal[0], Any, Literal[0], Any]]:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        always_partition: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class ClapAudioStage(GradientCheckpointingLayer):
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, downsample
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        always_partition: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class ClapAudioPatchMerging(nn.Module):
    def __init__(
        self, input_resolution: tuple[int], dim: int, norm_layer: nn.Module = ...
    ) -> None: ...
    def maybe_pad(self, input_feature, height, width):  # -> Tensor:
        ...
    def forward(
        self, input_feature: torch.Tensor, input_dimensions: tuple[int, int]
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ClapAudioEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def reshape_mel2img(self, normalized_input_features):  # -> Tensor:
        ...
    def forward(
        self,
        input_features,
        is_longer: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_hidden_states_before_downsampling: bool | None = ...,
        always_partition: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ClapAudioModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | ClapAudioModelOutput: ...

class ClapProjectionLayer(nn.Module):
    def __init__(self, config: ClapAudioConfig | ClapTextConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class ClapTextEmbeddings(nn.Module):
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

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class ClapTextSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class ClapTextSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ClapTextAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class ClapTextIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ClapTextOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ClapTextLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class ClapTextEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | BaseModelOutput: ...

class ClapTextPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class ClapPreTrainedModel(PreTrainedModel):
    config: ClapConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

class ClapAudioModel(ClapPreTrainedModel):
    config: ClapAudioConfig
    main_input_name = ...
    def __init__(self, config: ClapAudioConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None = ...,
        is_longer: torch.BoolTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class ClapTextModel(ClapPreTrainedModel):
    config: ClapTextConfig
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...

@auto_docstring
class ClapModel(ClapPreTrainedModel):
    config: ClapConfig
    def __init__(self, config: ClapConfig) -> None: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        is_longer: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        is_longer: torch.BoolTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ClapOutput: ...

@auto_docstring
class ClapTextModelWithProjection(ClapPreTrainedModel):
    config: ClapTextConfig
    def __init__(self, config: ClapTextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ClapTextModelOutput: ...

@auto_docstring
class ClapAudioModelWithProjection(ClapPreTrainedModel):
    config: ClapAudioConfig
    main_input_name = ...
    def __init__(self, config: ClapAudioConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None = ...,
        is_longer: torch.BoolTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ClapAudioModelOutput: ...

__all__ = [
    "ClapAudioModel",
    "ClapAudioModelWithProjection",
    "ClapModel",
    "ClapPreTrainedModel",
    "ClapTextModel",
    "ClapTextModelWithProjection",
]
