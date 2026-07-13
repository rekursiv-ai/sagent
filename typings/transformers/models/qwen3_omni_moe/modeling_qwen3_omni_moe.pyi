from dataclasses import dataclass

from torch import nn

import torch

from .configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import TransformersKwargs, check_model_inputs

@auto_docstring
class Qwen3OmniMoePreTrainedModel(PreTrainedModel):
    config: Qwen3OmniMoeConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...

class Qwen3OmniMoePreTrainedModelForConditionalGeneration(Qwen3OmniMoePreTrainedModel):
    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[torch.Tensor],
        grid_hs: list[torch.Tensor],
        grid_ws: list[torch.Tensor],
    ):  # -> Tensor:
        ...
    def get_chunked_index(
        self, token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
    ) -> list[tuple[int, int]]: ...
    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        use_audio_in_video: bool = ...,
        audio_seqlens: torch.LongTensor | None = ...,
        second_per_grids: torch.Tensor | None = ...,
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

class Qwen3OmniMoeAudioAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Qwen3OmniMoeAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=...) -> None: ...
    def forward(self, seqlen: int):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class Qwen3OmniMoeAudioEncoder(Qwen3OmniMoePreTrainedModel):
    config: Qwen3OmniMoeAudioEncoderConfig
    main_input_name = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value: nn.Module):  # -> None:
        ...
    @auto_docstring
    def forward(
        self, input_features, feature_lens=..., aftercnn_lens=...
    ):  # -> BaseModelOutput:
        ...
    def padded_and_mask_function(
        self, tensor_list, tensor_len, padding_value=..., padding_side=...
    ):  # -> tuple[Tensor, Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Qwen3OmniMoeVisionAttention(nn.Module):
    def __init__(self, config: Qwen3OmniMoeVisionEncoderConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Qwen3OmniMoeVisionPatchMerger(nn.Module):
    def __init__(
        self, config: Qwen3OmniMoeVisionEncoderConfig, use_postshuffle_norm=...
    ) -> None: ...
    def forward(self, hidden: torch.Tensor) -> torch.Tensor: ...

class Qwen3OmniMoeVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class Qwen3OmniMoeVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Qwen3OmniMoeVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, dim: int, theta: float = ...) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class Qwen3OmniMoeVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Qwen3OmniMoeVisionEncoder(Qwen3OmniMoePreTrainedModel):
    config: Qwen3OmniMoeVisionEncoderConfig
    _no_split_modules = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor: ...
    def fast_pos_embed_interpolate(self, grid_thw):  # -> Tensor:
        ...
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor: ...
    @property
    def deepstack_merger_list(self):  # -> ModuleList:
        ...

class Qwen3OmniMoeThinkerTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Qwen3OmniMoeTextConfig, device=...) -> None: ...
    def apply_interleaved_mrope(self, freqs, mrope_section): ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class Qwen3OmniMoeThinkerTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Qwen3OmniMoeThinkerTextSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3OmniMoeThinkerTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class Qwen3OmniMoeThinkerTextAttention(nn.Module):
    def __init__(self, config, layer_idx) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Qwen3OmniMoeThinkerTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor: ...

@auto_docstring
class Qwen3OmniMoeThinkerTextPreTrainedModel(PreTrainedModel):
    config = Qwen3OmniMoeTextConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...
    config_class = Qwen3OmniMoeTextConfig

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3OmniMoeTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

@auto_docstring(custom_intro=...)
class Qwen3OmniMoeThinkerTextModel(Qwen3OmniMoePreTrainedModel):
    config: Qwen3OmniMoeTextConfig
    _no_split_modules = ...
    config_class = Qwen3OmniMoeTextConfig
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTextConfig) -> None: ...
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
        visual_pos_masks: torch.Tensor | None = ...,
        deepstack_visual_embeds: list[torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

@dataclass
class Qwen3OmniMoeThinkerCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    rope_deltas: torch.LongTensor | None = ...

def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = ...,
    top_k=...,
    attention_mask: torch.Tensor | None = ...,
) -> torch.Tensor | int: ...

@auto_docstring(custom_intro=...)
class Qwen3OmniMoeThinkerForConditionalGeneration(
    Qwen3OmniMoePreTrainedModelForConditionalGeneration, GenerationMixin
):
    config: Qwen3OmniMoeThinkerConfig
    base_model_prefix = ...
    _tied_weights_keys = ...
    _no_split_modules = ...
    _can_record_outputs = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = ...,
        audio_feature_lengths: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = ...,
        video_features: torch.FloatTensor | None = ...,
    ):  # -> tuple[Tensor | Any, Tensor | Any, Tensor | Any]:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=...,
        input_features=...,
        pixel_values=...,
        pixel_values_videos=...,
        image_grid_thw=...,
        video_grid_thw=...,
        attention_mask=...,
        feature_attention_mask=...,
        audio_feature_lengths=...,
        position_ids=...,
        past_key_values=...,
        inputs_embeds=...,
        rope_deltas=...,
        labels=...,
        use_cache=...,
        output_router_logits: bool | None = ...,
        use_audio_in_video=...,
        cache_position=...,
        video_second_per_grid=...,
        **kwargs,
    ) -> tuple | Qwen3OmniMoeThinkerCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        pixel_values=...,
        pixel_values_videos=...,
        image_grid_thw=...,
        video_grid_thw=...,
        input_features=...,
        feature_attention_mask=...,
        use_audio_in_video=...,
        video_second_per_grid=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

class Qwen3OmniMoeTalkerResizeMLP(nn.Module):
    def __init__(self, config: Qwen3OmniMoeTalkerConfig) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

@dataclass
class Qwen3OmniMoeTalkerCodePredictorOutputWithPast(CausalLMOutputWithPast):
    generation_steps: int | None = ...

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3OmniMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self):  # -> str:
        ...

class Qwen3OmniMoeTalkerCodePredictorAttention(nn.Module):
    def __init__(self, config: Qwen3OmniMoeConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Qwen3OmniMoeMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Qwen3OmniMoeTalkerCodePredictorDecoderLayer(GradientCheckpointingLayer):
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

class Qwen3OmniMoeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Qwen3OmniMoeConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

@auto_docstring
class Qwen3OmniMoeTalkerCodePredictorModel(Qwen3OmniMoePreTrainedModel):
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = ...
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig) -> None: ...
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
    ) -> BaseModelOutputWithPast: ...
    def get_input_embeddings(self):  # -> ModuleList:
        ...

@auto_docstring
class Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(
    Qwen3OmniMoePreTrainedModel, GenerationMixin
):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = ...
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        position_ids=...,
        past_key_values=...,
        inputs_embeds=...,
        labels=...,
        use_cache=...,
        cache_position=...,
        generation_steps=...,
        **kwargs,
    ) -> CausalLMOutputWithPast: ...
    def get_input_embeddings(self):  # -> ModuleList:
        ...

@dataclass
class Qwen3OmniMoeTalkerOutputWithPast(MoeCausalLMOutputWithPast):
    generation_step: int | None = ...

class Qwen3OmniMoeTalkerRotaryEmbedding(Qwen3OmniMoeThinkerTextRotaryEmbedding): ...

class Qwen3OmniMoeTalkerTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Qwen3OmniMoeTalkerTextSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Qwen3OmniMoeTalkerDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor: ...

@auto_docstring(custom_intro=...)
class Qwen3OmniMoeTalkerModel(Qwen3OmniMoePreTrainedModel):
    config: Qwen3OmniMoeTextConfig
    _no_split_modules = ...
    config_class = Qwen3OmniMoeTalkerTextConfig
    base_model_prefix = ...
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTalkerTextConfig) -> None: ...
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
        visual_pos_masks: torch.Tensor | None = ...,
        deepstack_visual_embeds: list[torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...

@auto_docstring
class Qwen3OmniMoeTalkerForConditionalGeneration(
    Qwen3OmniMoeThinkerTextPreTrainedModel, GenerationMixin
):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    config_class = Qwen3OmniMoeTalkerConfig
    base_model_prefix = ...
    _no_split_modules = ...
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTalkerConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        use_audio_in_video=...,
        audio_feature_lengths=...,
        video_second_per_grid=...,
        image_grid_thw=...,
        video_grid_thw=...,
        position_ids=...,
        past_key_values=...,
        inputs_embeds=...,
        labels=...,
        use_cache=...,
        output_router_logits=...,
        cache_position=...,
        residual_codes=...,
        trailing_text_hidden=...,
        tts_pad_embed=...,
        generation_step=...,
        talker_input_ids=...,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast: ...
    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        use_audio_in_video: bool = ...,
        audio_seqlens: torch.LongTensor | None = ...,
        second_per_grids: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[torch.Tensor],
        grid_hs: list[torch.Tensor],
        grid_ws: list[torch.Tensor],
    ):  # -> Tensor:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

class Qwen3OmniMoeCausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=...,
        stride=...,
        groups=...,
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class Qwen3OmniMoeCausalTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=...) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class Qwen3OmniMoeConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None: ...
    def forward(self, hidden_states): ...

class Qwen3OmniMoeCode2WavRotatoryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Qwen3OmniMoeConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class Qwen3OmniMoeCode2WavAttention(nn.Module):
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Qwen3OmniMoeCode2WavMlp(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3OmniMoeCode2WavRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self):  # -> str:
        ...

class Qwen3OmniMoeCode2WavLayerScale(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x: torch.Tensor):  # -> Tensor:
        ...

class Qwen3OmniMoeCode2WavTransformerLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class Qwen3OmniMoeCode2WavTransformerModel(Qwen3OmniMoePreTrainedModel):
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        position_ids=...,
        past_key_values=...,
        inputs_embeds=...,
        use_cache=...,
        cache_position=...,
        **kwargs,
    ) -> BaseModelOutputWithPast: ...

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=...) -> None: ...
    def forward(self, hidden_states): ...

class Qwen3OmniMoeCode2WavDecoderResidualUnit(nn.Module):
    def __init__(self, dim: int = ..., dilation: int = ...) -> None: ...
    def forward(self, hidden_state): ...

class Qwen3OmniMoeCode2WavDecoderBlock(Qwen3OmniMoePreTrainedModel):
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx) -> None: ...
    def forward(self, hidden):  # -> Any:
        ...

class Qwen3OmniMoeCode2Wav(Qwen3OmniMoePreTrainedModel):
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig) -> None: ...
    def forward(self, codes):  # -> Any:
        ...
    def chunked_decode(
        self, codes, chunk_size=..., left_context_size=...
    ):  # -> Tensor:
        ...

class Qwen3OmniMoeForConditionalGeneration(
    Qwen3OmniMoePreTrainedModel, GenerationMixin
):
    config_class = Qwen3OmniMoeConfig
    def __init__(self, config: Qwen3OmniMoeConfig) -> None: ...
    def enable_talker(self):  # -> None:
        ...
    def disable_talker(self):  # -> None:
        ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = ...,
        speaker: str = ...,
        use_audio_in_video: bool = ...,
        return_audio: bool | None = ...,
        thinker_max_new_tokens: int = ...,
        thinker_eos_token_id: int = ...,
        talker_max_new_tokens: int = ...,
        talker_do_sample: bool = ...,
        talker_top_k: int = ...,
        talker_top_p: float = ...,
        talker_temperature: float = ...,
        talker_repetition_penalty: float = ...,
        **kwargs,
    ):  # -> tuple[GenerateOutput | LongTensor, None] | tuple[GenerateOutput | LongTensor, Tensor]:
        ...

__all__ = [
    "Qwen3OmniMoeCode2Wav",
    "Qwen3OmniMoeCode2WavDecoderBlock",
    "Qwen3OmniMoeCode2WavTransformerModel",
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen3OmniMoePreTrainedModel",
    ...,
    "Qwen3OmniMoeTalkerCodePredictorModel",
    ...,
    "Qwen3OmniMoeTalkerForConditionalGeneration",
    "Qwen3OmniMoeTalkerModel",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
    "Qwen3OmniMoeThinkerTextModel",
    "Qwen3OmniMoeThinkerTextPreTrainedModel",
]
