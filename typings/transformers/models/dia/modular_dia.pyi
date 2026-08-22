from typing import Any

from torch import nn

import torch

from .configuration_dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig
from .generation_dia import DiaGenerationMixin
from ..llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding
from ..phi3.modeling_phi3 import Phi3MLP
from ...cache_utils import EncoderDecoderCache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple

"""PyTorch Dia model."""
logger = ...

@auto_docstring
class DiaPreTrainedModel(PreTrainedModel):
    config: DiaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...
    main_input_name = ...
    _no_split_modules = ...

class DiaMultiChannelEmbedding(nn.Module):
    def __init__(self, config: DiaDecoderConfig) -> None: ...
    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class DiaMLP(Phi3MLP): ...
class DiaRMSNorm(LlamaRMSNorm): ...
class DiaRotaryEmbedding(LlamaRotaryEmbedding): ...

class DiaSelfAttention(LlamaAttention):
    def __init__(
        self,
        config: DiaEncoderConfig | DiaDecoderConfig,
        layer_idx: int,
        is_causal: bool = ...,
    ) -> None: ...

class DiaCrossAttention(nn.Module):
    def __init__(self, config: DiaDecoderConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class DiaEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaEncoderConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class DiaEncoder(DiaPreTrainedModel):
    def __init__(self, config: DiaEncoderConfig) -> None: ...
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput | tuple: ...

class DiaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaDecoderConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]: ...

class DiaDecoder(DiaPreTrainedModel):
    def __init__(self, config: DiaDecoderConfig) -> None: ...
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions | tuple: ...

@auto_docstring(custom_intro=...)
class DiaModel(DiaPreTrainedModel):
    def __init__(self, config: DiaConfig) -> None: ...
    def get_encoder(self):  # -> DiaEncoder:
        ...
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_position_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: BaseModelOutput | tuple | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class DiaForConditionalGeneration(DiaPreTrainedModel, DiaGenerationMixin):
    base_model_prefix = ...
    def __init__(self, config: DiaConfig) -> None: ...
    def get_encoder(self):  # -> DiaEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | DiaModel:
        ...
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_position_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: BaseModelOutput | tuple | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple | Seq2SeqLMOutput: ...

__all__ = ["DiaForConditionalGeneration", "DiaModel", "DiaPreTrainedModel"]
