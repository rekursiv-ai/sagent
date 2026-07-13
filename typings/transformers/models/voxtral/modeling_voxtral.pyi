from torch import nn

import torch

from .configuration_voxtral import VoxtralConfig, VoxtralEncoderConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs

logger = ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = ...,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class VoxtralAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        layer_idx: int | None = ...,
        config: VoxtralConfig | None = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class VoxtralEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: VoxtralConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = ...,
    ) -> torch.Tensor: ...

@auto_docstring
class VoxtralPreTrainedModel(PreTrainedModel):
    config: VoxtralConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _supports_cache_class = ...
    _supports_attention_backend = ...
    _can_compile_fullgraph = ...

@auto_docstring(custom_intro=...)
class VoxtralEncoder(VoxtralPreTrainedModel):
    config: VoxtralEncoderConfig
    main_input_name = ...
    _no_split_modules = ...
    _can_record_outputs = ...
    def __init__(self, config: VoxtralEncoderConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value: nn.Module):  # -> None:
        ...
    @check_model_inputs
    def forward(
        self, input_features, attention_mask=..., **kwargs: Unpack[TransformersKwargs]
    ):  # -> BaseModelOutput:
        ...

class VoxtralMultiModalProjector(nn.Module):
    def __init__(self, config: VoxtralConfig) -> None: ...
    def forward(self, audio_features):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class VoxtralForConditionalGeneration(VoxtralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    _keep_in_fp32_modules_strict = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Any:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_audio_features(self, input_features: torch.FloatTensor):  # -> Any:
        ...
    def get_audio_embeds(self, input_features: torch.FloatTensor):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
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
    def prepare_inputs_for_generation(self, *args, **kwargs):  # -> dict[Any, Any]:
        ...

__all__ = [
    "VoxtralEncoder",
    "VoxtralForConditionalGeneration",
    "VoxtralPreTrainedModel",
]
