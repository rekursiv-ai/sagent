from typing import Any
from torch import nn

import torch

from .configuration_dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig
from .generation_dia import DiaGenerationMixin
from ...cache_utils import Cache, EncoderDecoderCache
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from ...utils.deprecation import deprecate_kwarg

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

class DiaMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

@use_kernel_forward_from_hub("RMSNorm")
class DiaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class DiaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: DiaConfig, device=...) -> None: ...
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
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...

class DiaSelfAttention(nn.Module):
    def __init__(
        self,
        config: DiaEncoderConfig | DiaDecoderConfig,
        layer_idx: int,
        is_causal: bool = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

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
