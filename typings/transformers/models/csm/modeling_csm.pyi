from dataclasses import dataclass

from torch import nn
from transformers.utils.generic import check_model_inputs

import torch

from .configuration_csm import CsmConfig, CsmDepthDecoderConfig
from .generation_csm import CsmGenerationMixin
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class CsmOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    depth_decoder_loss: torch.FloatTensor | None = ...
    depth_decoder_logits: torch.FloatTensor | None = ...
    depth_decoder_past_key_values: Cache | None = ...
    depth_decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    depth_decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    backbone_loss: torch.FloatTensor | None = ...

@use_kernel_forward_from_hub("RMSNorm")
class CsmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class CsmRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: CsmConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class CsmMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
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

class CsmAttention(nn.Module):
    def __init__(self, config: CsmConfig, layer_idx: int) -> None: ...
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

class CsmDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: CsmConfig, layer_idx: int) -> None: ...
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
@auto_docstring
class CsmPreTrainedModel(PreTrainedModel):
    config: CsmConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

@auto_docstring
class CsmDepthDecoderModel(CsmPreTrainedModel):
    config: CsmDepthDecoderConfig
    def __init__(self, config) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        backbone_last_hidden_state: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

class CsmCodebooksHead(nn.Module):
    def __init__(self, hidden_size, num_codebooks, vocab_size) -> None: ...
    def forward(self, hidden_states, cache_position=...):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class CsmDepthDecoderForCausalLM(CsmPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        backbone_last_hidden_state: torch.FloatTensor | None = ...,
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
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

class CsmBackboneModelEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_ids):  # -> Any:
        ...

@auto_docstring
class CsmBackboneModel(CsmPreTrainedModel):
    def __init__(self, config) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class CsmForConditionalGeneration(CsmPreTrainedModel, CsmGenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> CsmBackboneModelEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # -> tuple[Any | Self, Any] | Self:
        ...
    def save_pretrained(self, *args, **kwargs):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_values: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        input_values_cutoffs: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CsmOutputWithPast: ...

__all__ = [
    "CsmBackboneModel",
    "CsmDepthDecoderForCausalLM",
    "CsmDepthDecoderModel",
    "CsmForConditionalGeneration",
    "CsmPreTrainedModel",
]
