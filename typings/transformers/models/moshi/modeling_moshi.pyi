from dataclasses import dataclass

from torch import nn

import torch

from .configuration_moshi import MoshiConfig, MoshiDepthConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput, Seq2SeqLMOutput
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Moshi model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class MoshiConditionalGenerationGenerateOutput(ModelOutput):
    audio_sequences: torch.Tensor | None = ...
    sequences: torch.LongTensor | None = ...
    sequences_scores: torch.FloatTensor | None = ...
    scores: tuple[torch.FloatTensor] | None = ...
    logits: tuple[torch.FloatTensor] | None = ...
    beam_indices: torch.LongTensor | None = ...
    attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = ...
    past_key_values: Cache | None = ...
    audio_codes: torch.LongTensor | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    `MoshiForCausalLM` outputs.
    """
)
class MoshiCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    `MoshiForConditionalGeneration` outputs.
    """
)
class MoshiConditionalGenerationOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    depth_loss: torch.FloatTensor | None = ...
    audio_logits: torch.FloatTensor | None = ...
    depth_past_key_values: Cache | None = ...
    depth_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    depth_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring
class MoshiUnconditionalInput(ModelOutput):
    input_ids: torch.LongTensor | None = ...
    user_audio_codes: torch.Tensor | None = ...
    moshi_audio_codes: torch.Tensor | None = ...
    attention_mask: torch.LongTensor | None = ...

class MoshiRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = ...) -> None: ...
    def forward(self, x): ...
    def extra_repr(self):  # -> str:
        ...

class MoshiFlexibleLinear(nn.Module):
    def __init__(self, input_size, output_size, num_layers) -> None: ...
    def forward(self, x, layer_idx=...):  # -> Tensor:
        ...

class MoshiLinear(nn.Module):
    def __init__(
        self, input_dim, output_dim, num_codebooks, use_flexible_linear=...
    ) -> None: ...
    def forward(self, x, layer_idx=...):  # -> Any:
        ...

class MoshiRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: MoshiConfig, device=...) -> None: ...
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

class MoshiGatingMLP(nn.Module):
    def __init__(self, config, use_flexible_linear=...) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, layer_idx: int | None = ...
    ) -> torch.Tensor: ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...

class MoshiAttention(nn.Module):
    def __init__(
        self,
        config: MoshiConfig,
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

class MoshiFlashAttention2(MoshiAttention):
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

class MoshiSdpaAttention(MoshiAttention):
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

MOSHI_ATTENTION_CLASSES = ...

class MoshiDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: MoshiConfig,
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
class MoshiPreTrainedModel(PreTrainedModel):
    config: MoshiConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    main_input_name = ...

class MoshiDepthDecoder(MoshiPreTrainedModel, GenerationMixin):
    config: MoshiDepthConfig
    def __init__(self, config: MoshiDepthConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        last_hidden_state: torch.LongTensor | None = ...,
        attention_mask: torch.BoolTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring
class MoshiModel(MoshiPreTrainedModel):
    def __init__(self, config: MoshiConfig) -> None: ...
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

@auto_docstring(custom_intro=...)
class MoshiForCausalLM(MoshiPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
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
        labels: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | MoshiCausalLMOutputWithPast: ...

@auto_docstring(custom_intro=...)
class MoshiForConditionalGeneration(MoshiPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    config: MoshiConfig
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    def __init__(self, config: MoshiConfig) -> None: ...
    def get_audio_encoder(self):  # -> Any:
        ...
    def get_depth_decoder(self):  # -> MoshiDepthDecoder:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.BoolTensor | None = ...,
        user_input_values: torch.FloatTensor | None = ...,
        user_audio_codes: torch.Tensor | None = ...,
        moshi_input_values: torch.FloatTensor | None = ...,
        moshi_audio_codes: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        text_labels: torch.LongTensor | None = ...,
        audio_labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | Seq2SeqLMOutput: ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = ...,
        user_input_values: torch.FloatTensor | None = ...,
        user_audio_codes: torch.Tensor | None = ...,
        moshi_input_values: torch.FloatTensor | None = ...,
        moshi_audio_codes: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        return_audio_waveforms: bool | None = ...,
        return_audio_codes: bool | None = ...,
        concat_unconditional_inputs: bool | None = ...,
        **kwargs,
    ) -> torch.LongTensor: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        logits_to_keep=...,
        user_delay_pattern_mask=...,
        moshi_delay_pattern_mask=...,
        kwargs_depth_decoder=...,
        blank_user_audio_codes: torch.FloatTensor | None = ...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> None:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def freeze_audio_encoder(self):  # -> None:
        ...
    def freeze_depth_decoder(self):  # -> None:
        ...
    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):  # -> Tensor:
        ...
    def build_delay_pattern_mask(
        self,
        input_ids: torch.LongTensor,
        bos_token_id: int,
        pad_token_id: int,
        max_length: int | None = ...,
    ):  # -> tuple[LongTensor, Tensor]:
        ...
    def get_unconditional_inputs(self, num_samples=...):  # -> MoshiUnconditionalInput:
        ...

__all__ = [
    "MoshiForCausalLM",
    "MoshiForConditionalGeneration",
    "MoshiModel",
    "MoshiPreTrainedModel",
]
