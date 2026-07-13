from dataclasses import dataclass

from torch import nn

import torch

from .configuration_parakeet import ParakeetCTCConfig, ParakeetEncoderConfig
from ..fastspeech2_conformer.modeling_fastspeech2_conformer import (
    FastSpeech2ConformerConvolutionModule,
)
from ..llama.modeling_llama import LlamaAttention
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs

"""PyTorch Parakeet model."""

class ParakeetEncoderRelPositionalEncoding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: ParakeetEncoderConfig, device=...) -> None: ...
    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class ParakeetEncoderFeedForward(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class ParakeetEncoderConvolutionModule(FastSpeech2ConformerConvolutionModule):
    def __init__(self, config: ParakeetEncoderConfig, module_config=...) -> None: ...

class ParakeetEncoderAttention(LlamaAttention):
    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class ParakeetEncoderSubsamplingConv2D(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None: ...
    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor = ...
    ):  # -> Any:
        ...

class ParakeetEncoderBlock(GradientCheckpointingLayer):
    def __init__(
        self, config: ParakeetEncoderConfig, layer_idx: int | None = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_embeddings: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

@auto_docstring
class ParakeetPreTrainedModel(PreTrainedModel):
    config: ParakeetCTCConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flat_attention_mask = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _supports_flash_attn = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

@auto_docstring(custom_intro=...)
class ParakeetEncoder(ParakeetPreTrainedModel):
    config: ParakeetEncoderConfig
    base_model_prefix = ...
    def __init__(self, config: ParakeetEncoderConfig) -> None: ...
    @auto_docstring
    @check_model_inputs
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

@dataclass
class ParakeetGenerateOutput(ModelOutput):
    sequences: torch.LongTensor
    logits: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = ...

@auto_docstring(custom_intro=...)
class ParakeetForCTC(ParakeetPreTrainedModel):
    config: ParakeetCTCConfig
    def __init__(self, config: ParakeetCTCConfig) -> None: ...
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput: ...
    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        return_dict_in_generate: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ParakeetGenerateOutput | torch.LongTensor: ...

__all__ = ["ParakeetEncoder", "ParakeetForCTC", "ParakeetPreTrainedModel"]
