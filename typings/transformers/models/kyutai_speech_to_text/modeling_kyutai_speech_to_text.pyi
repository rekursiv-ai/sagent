from typing import Any

from torch import nn

import torch

from .configuration_kyutai_speech_to_text import KyutaiSpeechToTextConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
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

class KyutaiSpeechToTextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = ...) -> None: ...
    def forward(self, x): ...
    def extra_repr(self):  # -> str:
        ...

class KyutaiSpeechToTextFlexibleLinear(nn.Module):
    def __init__(self, input_size, output_size, num_layers) -> None: ...
    def forward(self, x, layer_idx=...):  # -> Tensor:
        ...

@auto_docstring
class KyutaiSpeechToTextPreTrainedModel(PreTrainedModel):
    config: KyutaiSpeechToTextConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    main_input_name = ...

class KyutaiSpeechToTextConv1dPaddingCache:
    def __init__(
        self,
        num_layers: int,
        per_layer_padding: list[int],
        per_layer_padding_mode: list[str],
        per_layer_in_channels: list[int],
    ) -> None: ...
    def update(self, hidden_states: torch.Tensor, layer_idx: int):  # -> Tensor | None:
        ...

class KyutaiSpeechToTextEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_ids):  # -> Any:
        ...

class KyutaiSpeechToTextLinear(nn.Module):
    def __init__(
        self, input_dim, output_dim, num_codebooks, use_flexible_linear=...
    ) -> None: ...
    def forward(self, x, layer_idx=...):  # -> Any:
        ...

class KyutaiSpeechToTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: KyutaiSpeechToTextConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class KyutaiSpeechToTextGatingMLP(nn.Module):
    def __init__(self, config, use_flexible_linear=...) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, layer_idx: int | None = ...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...

class KyutaiSpeechToTextAttention(nn.Module):
    def __init__(
        self,
        config: KyutaiSpeechToTextConfig,
        layer_idx: int | None = ...,
        use_flexible_linear=...,
        use_rope=...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class KyutaiSpeechToTextFlashAttention2(KyutaiSpeechToTextAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class KyutaiSpeechToTextSdpaAttention(KyutaiSpeechToTextAttention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

KYUTAI_SPEECH_TO_TEXT_ATTENTION_CLASSES = ...

class KyutaiSpeechToTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: KyutaiSpeechToTextConfig,
        layer_idx: int,
        use_flexible_linear: bool,
        use_rope=...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class KyutaiSpeechToTextModel(KyutaiSpeechToTextPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring
class KyutaiSpeechToTextForConditionalGeneration(
    KyutaiSpeechToTextPreTrainedModel, GenerationMixin
):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    _keep_in_fp32_modules_strict = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        *args,
        audio_tokens: torch.LongTensor | None = ...,
        input_values: torch.FloatTensor | None = ...,
        padding_mask: torch.Tensor | None = ...,
        audio_window_size: int | None = ...,
        current_window: tuple[int, int] | None = ...,
        encoder_past_key_values: Cache | None = ...,
        padding_cache: KyutaiSpeechToTextConv1dPaddingCache | None = ...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # -> tuple[Any | Self, Any] | Self:
        ...
    def save_pretrained(self, *args, **kwargs):  # -> None:
        ...
    def generate(self, *args, **kwargs):  # -> GenerateOutput | LongTensor:
        ...

__all__ = [
    "KyutaiSpeechToTextForConditionalGeneration",
    "KyutaiSpeechToTextModel",
    "KyutaiSpeechToTextPreTrainedModel",
]
