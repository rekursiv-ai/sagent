from typing import Any
from dataclasses import dataclass

from torch import nn
from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig

import torch

from .configuration_llama4 import Llama4Config, Llama4TextConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

logger = ...

class Llama4TextExperts(nn.Module):
    def __init__(self, config: Llama4TextConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Llama4TextMLP(nn.Module):
    def __init__(self, config, intermediate_size=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Llama4TextL2Norm(torch.nn.Module):
    def __init__(self, eps: float = ...) -> None: ...
    def forward(self, x): ...
    def extra_repr(self):  # -> str:
        ...

class Llama4TextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, x): ...
    def extra_repr(self):  # -> str:
        ...

class Llama4Router(nn.Linear):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Tensor, Tensor]:
        ...

@use_kernel_forward_from_hub("Llama4TextMoe")
class Llama4TextMoe(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class Llama4TextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Llama4TextConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> Tensor:
        ...

def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
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
def vision_eager_attention_forward(
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

class Llama4TextAttention(nn.Module):
    def __init__(self, config: Llama4TextConfig, layer_idx) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Llama4TextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class Llama4PreTrainedModel(PreTrainedModel):
    config: Llama4Config
    supports_gradient_checkpointing = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...

@auto_docstring
class Llama4TextModel(Llama4PreTrainedModel):
    _no_split_modules = ...
    base_model_prefix = ...
    config: Llama4TextConfig
    _can_record_outputs = ...
    def __init__(self, config: Llama4TextConfig) -> None: ...
    @can_return_tuple
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

class Llama4ForCausalLM(Llama4PreTrainedModel, GenerationMixin):
    _no_split_modules = ...
    base_model_prefix = ...
    _tied_weights_keys = ...
    _tp_plan = ...
    config: Llama4TextConfig
    def __init__(self, config: Llama4TextConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast: ...

@dataclass
@auto_docstring(custom_intro=...)
class Llama4CausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: torch.FloatTensor | None = ...

class Llama4VisionMLP2(torch.nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Llama4MultiModalProjector(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, image_features):  # -> Any:
        ...

def pixel_shuffle(input_tensor, shuffle_ratio): ...

class Llama4VisionPixelShuffleMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def reshape_for_broadcast(freqs_ci: torch.Tensor, query: torch.Tensor):  # -> Tensor:
    ...
def vision_apply_rotary_emb(
    query: torch.Tensor, key: torch.Tensor, freqs_ci: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Llama4VisionAttention(nn.Module):
    def __init__(self, config: Llama4VisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Llama4VisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Llama4VisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Llama4VisionConfig) -> None: ...
    def forward(
        self,
        hidden_state: torch.Tensor,
        freqs_ci: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class Llama4VisionEncoder(nn.Module):
    def __init__(self, config: Llama4VisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

class Llama4UnfoldConvolution(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Llama4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class Llama4VisionModel(Llama4PreTrainedModel):
    base_model_prefix = ...
    _no_split_modules = ...
    config: Llama4VisionConfig
    def __init__(self, config: Llama4VisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> Llama4UnfoldConvolution:
        ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]: ...

class Llama4ForConditionalGeneration(Llama4PreTrainedModel, GenerationMixin):
    _no_split_modules = ...
    _tp_plan = ...
    base_model_prefix = ...
    config: Llama4Config
    def __init__(self, config: Llama4Config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> None:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> PreTrainedModel | Llama4ForCausalLM:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_select_strategy: str,
        **kwargs,
    ):  # -> Any:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Any:
        ...
    @auto_docstring
    @deprecate_kwarg("vision_feature_layer", version="4.58")
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Llama4CausalLMOutputWithPast: ...
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

__all__ = [
    "Llama4ForCausalLM",
    "Llama4ForConditionalGeneration",
    "Llama4PreTrainedModel",
    "Llama4TextModel",
    "Llama4VisionModel",
]
