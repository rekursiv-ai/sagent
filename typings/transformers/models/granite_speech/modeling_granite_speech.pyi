from dataclasses import dataclass

from torch import nn

import torch

from .configuration_granite_speech import (
    GraniteSpeechConfig,
    GraniteSpeechEncoderConfig,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class GraniteSpeechCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class GraniteSpeechEncoderProjector(nn.Module):
    def __init__(self, config: GraniteSpeechConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteSpeechConformerFeedForward(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteSpeechConformerAttention(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_dists: torch.Tensor
    ) -> torch.Tensor: ...

class GraniteSpeechConformerDepthWiseConv1d(nn.Module):
    def __init__(self, chan_in: int, chan_out: int, kernel_size: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteSpeechConformerConvModule(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteSpeechConformerBlock(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_dists: torch.Tensor
    ) -> torch.Tensor: ...

class GraniteSpeechCTCEncoder(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

@auto_docstring
class GraniteSpeechPreTrainedModel(PreTrainedModel):
    config: GraniteSpeechConfig
    _supports_flash_attn = ...
    _supports_sdpa = ...

@auto_docstring(custom_intro=...)
class GraniteSpeechForConditionalGeneration(
    GraniteSpeechPreTrainedModel, GenerationMixin
):
    def __init__(self, config: GraniteSpeechConfig) -> None: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def get_output_embeddings(self):  # -> Any:
        ...
    def get_audio_features(self, input_features: torch.Tensor) -> torch.Tensor: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        input_features_mask: torch.Tensor | None = ...,
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
        **lm_kwargs,
    ) -> tuple[torch.Tensor] | GraniteSpeechCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        input_features=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> Any:
        ...
    def get_merged_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def generate(self, *args, **kwargs) -> torch.LongTensor: ...
    def save_pretrained(self, save_directory, *args, **kwargs):  # -> None:
        ...

__all__ = [
    "GraniteSpeechCTCEncoder",
    "GraniteSpeechForConditionalGeneration",
    "GraniteSpeechPreTrainedModel",
]
