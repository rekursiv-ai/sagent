from dataclasses import dataclass

from torch import nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLMLP,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionBlock,
)
from transformers.models.qwen2_audio.configuration_qwen2_audio import (
    Qwen2AudioEncoderConfig,
)
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoderLayer
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding

import torch

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import ModelOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring

"""PyTorch Qwen2.5Omni model (Audio, Image, Video)."""
logger = ...

class Qwen2_5OmniVisionEncoderConfig(Qwen2_5_VLVisionConfig):
    model_type = ...
    def __init__(
        self,
        depth=...,
        hidden_size=...,
        hidden_act=...,
        intermediate_size=...,
        num_heads=...,
        in_channels=...,
        patch_size=...,
        spatial_merge_size=...,
        temporal_patch_size=...,
        window_size=...,
        out_hidden_size=...,
        fullatt_block_indexes=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5OmniAudioEncoderConfig(Qwen2AudioEncoderConfig):
    model_type = ...
    def __init__(
        self,
        num_mel_bins=...,
        encoder_layers=...,
        encoder_attention_heads=...,
        encoder_ffn_dim=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_function=...,
        activation_dropout=...,
        scale_embedding=...,
        initializer_range=...,
        max_source_positions=...,
        n_window=...,
        output_dim=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5OmniTextConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        use_sliding_window=...,
        sliding_window=...,
        max_window_layers=...,
        layer_types=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5OmniThinkerConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        audio_config=...,
        vision_config=...,
        text_config=...,
        audio_token_index=...,
        image_token_index=...,
        video_token_index=...,
        position_id_per_seconds=...,
        seconds_per_chunk=...,
        audio_start_token_id=...,
        audio_end_token_id=...,
        user_token_id=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5OmniTalkerConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        audio_token_index=...,
        image_token_index=...,
        video_token_index=...,
        vocab_size=...,
        tts_text_start_token_id=...,
        tts_text_end_token_id=...,
        tts_text_pad_token_id=...,
        tts_codec_start_token_id=...,
        tts_codec_end_token_id=...,
        tts_codec_pad_token_id=...,
        tts_codec_mask_token_id=...,
        vision_start_token_id=...,
        vision_end_token_id=...,
        embedding_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        rms_norm_eps=...,
        head_dim=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        use_sliding_window=...,
        sliding_window=...,
        max_window_layers=...,
        attention_dropout=...,
        rope_scaling=...,
        position_id_per_seconds=...,
        seconds_per_chunk=...,
        audio_start_token_id=...,
        audio_end_token_id=...,
        initializer_range=...,
        spatial_merge_size=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5OmniDiTConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        ff_mult=...,
        emb_dim=...,
        head_dim=...,
        rope_theta=...,
        max_position_embeddings=...,
        block_size=...,
        look_ahead_layers=...,
        look_backward_layers=...,
        repeats=...,
        num_embeds=...,
        mel_dim=...,
        dropout=...,
        enc_emb_dim=...,
        enc_dim=...,
        enc_channels=...,
        enc_kernel_sizes=...,
        enc_dilations=...,
        enc_attention_channels=...,
        enc_res2net_scale=...,
        enc_se_channels=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5OmniBigVGANConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        mel_dim=...,
        upsample_initial_channel=...,
        resblock_kernel_sizes=...,
        resblock_dilation_sizes=...,
        upsample_rates=...,
        upsample_kernel_sizes=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5OmniToken2WavConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(self, dit_config=..., bigvgan_config=..., **kwargs) -> None: ...

class Qwen2_5OmniConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        thinker_config=...,
        talker_config=...,
        token2wav_config=...,
        enable_audio_output: bool = ...,
        **kwargs,
    ) -> None: ...
    def get_text_config(self, *args, **kwargs):  # -> PretrainedConfig:
        ...

class Qwen2_5OmniPreTrainedModel(Qwen2_5_VLPreTrainedModel):
    config: Qwen2_5OmniConfig
    _can_compile_fullgraph = ...

class Qwen2_5OmniPreTrainedModelForConditionalGeneration(Qwen2_5OmniPreTrainedModel):
    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[int],
        grid_hs: list[int],
        grid_ws: list[int],
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

@dataclass
@auto_docstring(custom_intro=...)
class Qwen2_5OmniThinkerCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    rope_deltas: torch.LongTensor | None = ...

class Qwen2_5OmniAudioAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Qwen2_5OmniAudioEncoderLayer(Qwen2AudioEncoderLayer):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig) -> None: ...
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
class Qwen2_5OmniAudioEncoder(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniAudioEncoderConfig
    main_input_name = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value: nn.Module):  # -> None:
        ...
    @auto_docstring
    def forward(
        self, input_features, feature_lens=..., aftercnn_lens=..., **kwargs
    ):  # -> BaseModelOutput:
        ...
    def padded_and_mask_function(
        self, tensor_list, tensor_len, padding_value=..., padding_side=...
    ):  # -> tuple[Tensor, Tensor, Tensor]:
        ...

def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor: ...

class Qwen2_5OmniVisionAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Qwen2_5OmniVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Qwen2_5OmniVisionEncoder(Qwen2_5_VisionTransformerPretrainedModel):
    config: Qwen2_5OmniVisionEncoderConfig
    _no_split_modules = ...
    def __init__(
        self, config: Qwen2_5OmniVisionEncoderConfig, *inputs, **kwargs
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor: ...

class Qwen2_5OmniRotaryEmbedding(Qwen2VLRotaryEmbedding):
    def __init__(self, config: Qwen2_5OmniThinkerConfig, device=...) -> None: ...

class Qwen2_5OmniAttention(Qwen2_5_VLAttention):
    def __init__(
        self, config: Qwen2_5OmniConfig, layer_idx: int | None = ...
    ) -> None: ...

class Qwen2MLP(Qwen2_5_VLMLP): ...

class Qwen2_5OmniThinkerTextModel(Qwen2_5_VLTextModel):
    config: Qwen2_5OmniTextConfig
    _no_split_modules = ...
    def __init__(self, config: Qwen2_5OmniTextConfig) -> None: ...

@auto_docstring(custom_intro=...)
class Qwen2_5OmniThinkerForConditionalGeneration(
    Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin
):
    config: Qwen2_5OmniThinkerConfig
    base_model_prefix = ...
    _tied_weights_keys = ...
    _no_split_modules = ...
    def __init__(self, config: Qwen2_5OmniThinkerConfig) -> None: ...
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
    ):  # -> tuple[Any, Any, Any]:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        feature_attention_mask: torch.Tensor | None = ...,
        audio_feature_lengths: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        rope_deltas: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        use_audio_in_video: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        video_second_per_grid: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2_5OmniThinkerCausalLMOutputWithPast: ...
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

@dataclass
@auto_docstring(custom_intro=...)
class Qwen2_5OmniTalkerCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    rope_deltas: torch.LongTensor | None = ...
    thinker_reply_part: torch.FloatTensor | None = ...

class Qwen2_5OmniTalkerModel(Qwen2_5_VLTextModel):
    config: Qwen2_5OmniTalkerConfig
    _no_split_modules = ...
    def __init__(self, config: Qwen2_5OmniTalkerConfig) -> None: ...

class Qwen2_5OmniTalkerForConditionalGeneration(
    Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin
):
    config: Qwen2_5OmniTalkerConfig
    base_model_prefix = ...
    def __init__(self, config: Qwen2_5OmniTalkerConfig) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        thinker_reply_part: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        rope_deltas: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        input_text_ids: torch.LongTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        use_audio_in_video: bool | None = ...,
        audio_feature_lengths: torch.LongTensor | None = ...,
        video_second_per_grid: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Qwen2_5OmniTalkerCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_text_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        thinker_reply_part=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        pixel_values=...,
        pixel_values_videos=...,
        image_grid_thw=...,
        video_grid_thw=...,
        input_audio_features=...,
        audio_feature_attention_mask=...,
        audio_feature_lengths=...,
        use_audio_in_video=...,
        video_second_per_grid=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

class Qwen2_5OmniDiTRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, dim, base=...) -> None: ...
    def forward(self, x):  # -> tuple[Tensor, Tensor]:
        ...

def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class TimeDelayNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Any:
        ...

class Res2NetBlock(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, scale=..., kernel_size=..., dilation=...
    ) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels) -> None: ...
    def forward(self, hidden_states): ...

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=...) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=...,
        se_channels=...,
        kernel_size=...,
        dilation=...,
    ) -> None: ...
    def forward(self, hidden_state): ...

class ECAPA_TimeDelayNet(torch.nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class DiTInputEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        condition_vector: torch.Tensor,
        code_embed: torch.Tensor,
        drop_audio_cond: bool | None = ...,
        code_embed_uncond: bool | None = ...,
        apply_cfg: bool | None = ...,
    ):  # -> Tensor:
        ...

class DiTCodecEmbedding(nn.Module):
    def __init__(self, codec_num_embeds, codec_dim, repeats) -> None: ...
    def forward(self, code, drop_code=...):  # -> Tensor:
        ...

class Qwen2_5_OmniAdaLayerNormZero(nn.Module):
    def __init__(self, dim) -> None: ...
    def forward(
        self, hidden_states, emb=...
    ):  # -> tuple[Any, Tensor, Tensor, Tensor, Tensor]:
        ...

class Qwen2_5_OmniAdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim) -> None: ...
    def forward(self, hidden_states, emb):  # -> Any:
        ...

class DiTMLP(nn.Module):
    def __init__(self, dim, mult=..., dropout=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class DiTAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig) -> None: ...
    def forward(
        self, hidden_states, position_embeddings=..., attention_mask=...
    ) -> torch.Tensor: ...

class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim) -> None: ...
    def forward(self, hidden_states, scale=...):  # -> Tensor:
        ...

class DiTTimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=...) -> None: ...
    def forward(self, timestep):  # -> Any:
        ...

class DiTDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2_5OmniDiTConfig,
        look_ahead_block=...,
        look_backward_block=...,
    ) -> None: ...
    def forward(
        self, hidden_states, timestep, position_embeddings=..., block_diff=...
    ): ...

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=...) -> None: ...
    def forward(self, hidden_states): ...

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):  # -> Tensor:
    ...

class UpSample1d(nn.Module):
    def __init__(self, ratio=..., kernel_size=...) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class DownSample1d(nn.Module):
    def __init__(self, ratio=..., kernel_size=...) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class TorchActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = ...,
        down_ratio: int = ...,
        up_kernel_size: int = ...,
        down_kernel_size: int = ...,
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class AMPBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=..., dilation=...) -> None: ...
    def forward(self, hidden_states): ...

@auto_docstring(custom_intro=...)
class Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniBigVGANConfig
    def __init__(self, config: Qwen2_5OmniBigVGANConfig) -> None: ...
    def normalize_spectrogram(self, spectrogram, max_value, min_db):  # -> Tensor:
        ...
    def amplitude_to_db(self, amplitude, min_db_level):  # -> Tensor:
        ...
    def process_mel_spectrogram(self, mel_spectrogram):  # -> Tensor:
        ...
    def forward(self, mel_spectrogram):  # -> Tensor:
        ...

class RungeKutta4ODESolver:
    def __init__(self, function, initial_value) -> None: ...
    def integrate(self, time_points):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class Qwen2_5OmniToken2WavDiTModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniDiTConfig
    _no_split_modules = ...
    def __init__(self, config: Qwen2_5OmniDiTConfig) -> None: ...
    def forward(
        self,
        hidden_states,
        condition_vector,
        speaker_embedding,
        quantized_code,
        time_step,
        drop_audio_conditioning=...,
        drop_code=...,
        apply_cfg=...,
    ):  # -> Any:
        ...
    @torch.no_grad()
    def sample(
        self,
        conditioning_vector,
        reference_mel_spectrogram,
        quantized_code,
        num_steps=...,
        guidance_scale=...,
        sway_coefficient=...,
    ):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class Qwen2_5OmniToken2WavModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniToken2WavConfig
    base_model_prefix = ...
    _no_split_modules = ...
    def __init__(self, config: Qwen2_5OmniToken2WavConfig) -> None: ...
    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=...,
        guidance_scale=...,
        sway_coefficient=...,
        **kwargs,
    ):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class Qwen2_5OmniForConditionalGeneration(Qwen2_5OmniPreTrainedModel, GenerationMixin):
    config: Qwen2_5OmniConfig
    _no_split_modules = ...
    def __init__(self, config) -> None: ...
    def enable_talker(self):  # -> None:
        ...
    def load_speakers(self, path):  # -> None:
        ...
    def disable_talker(self):  # -> None:
        ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=...,
        cache_dir=...,
        ignore_mismatched_sizes=...,
        force_download=...,
        local_files_only=...,
        token=...,
        revision=...,
        use_safetensors=...,
        weights_only=...,
        **kwargs,
    ):  # -> Self:
        ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = ...,
        speaker: str = ...,
        use_audio_in_video: bool = ...,
        return_audio: bool | None = ...,
        thinker_max_new_tokens: int = ...,
        talker_max_new_tokens: int = ...,
        talker_do_sample: bool = ...,
        talker_top_k: int = ...,
        talker_top_p: float = ...,
        talker_temperature: float = ...,
        talker_eos_token_id: list[int] = ...,
        talker_repetition_penalty: float = ...,
        **kwargs,
    ):  # -> GenerateOutput | LongTensor | tuple[LongTensor | Any, Any]:
        ...

__all__ = [
    "Qwen2_5OmniConfig",
    "Qwen2_5OmniForConditionalGeneration",
    "Qwen2_5OmniPreTrainedModel",
    "Qwen2_5OmniPreTrainedModelForConditionalGeneration",
    "Qwen2_5OmniTalkerConfig",
    "Qwen2_5OmniTalkerForConditionalGeneration",
    "Qwen2_5OmniTalkerModel",
    "Qwen2_5OmniThinkerConfig",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen2_5OmniThinkerTextModel",
    "Qwen2_5OmniToken2WavBigVGANModel",
    "Qwen2_5OmniToken2WavConfig",
    "Qwen2_5OmniToken2WavDiTModel",
    "Qwen2_5OmniToken2WavModel",
]
