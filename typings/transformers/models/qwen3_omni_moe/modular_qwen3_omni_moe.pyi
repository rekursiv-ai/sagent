from dataclasses import dataclass

from torch import nn

import torch

from ..mimi.modeling_mimi import MimiLayerScale
from ..qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniThinkerConfig,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioAttention,
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniPreTrainedModel,
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    SnakeBeta,
)
from ..qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniProcessorKwargs,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from ..qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoePreTrainedModel,
    Qwen3MoeSparseMoeBlock,
)
from ..qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeVisionConfig
from ..qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextModel,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeVisionAttention,
    Qwen3VLMoeVisionModel,
)
from ...audio_utils import AudioInput
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...image_utils import ImageInput
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from ...processing_utils import ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, can_return_tuple
from ...utils.generic import TransformersKwargs, check_model_inputs
from ...video_utils import VideoInput

"""PyTorch Qwen3Omni model (Audio, Image, Video)."""
logger = ...

class Qwen3OmniMoeAudioEncoderConfig(Qwen2_5OmniAudioEncoderConfig):
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
        n_window_infer=...,
        conv_chunksize=...,
        downsample_hidden_size=...,
        **kwargs,
    ) -> None: ...

class Qwen3OmniMoeVisionEncoderConfig(Qwen3VLMoeVisionConfig): ...

class Qwen3OmniMoeTextConfig(Qwen3MoeConfig):
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
        attention_bias=...,
        sliding_window=...,
        attention_dropout=...,
        decoder_sparse_step=...,
        moe_intermediate_size=...,
        num_experts_per_tok=...,
        num_experts=...,
        norm_topk_prob=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        mlp_only_layers=...,
        **kwargs,
    ) -> None: ...

class Qwen3OmniMoeThinkerConfig(Qwen2_5OmniThinkerConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        audio_config=...,
        vision_config=...,
        text_config=...,
        audio_token_id=...,
        image_token_id=...,
        video_token_id=...,
        position_id_per_seconds=...,
        audio_start_token_id=...,
        user_token_id=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Qwen3OmniMoeTalkerCodePredictorConfig(Qwen3Config):
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        head_dim=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        sliding_window=...,
        layer_types=...,
        attention_dropout=...,
        num_code_groups=...,
        **kwargs,
    ) -> None: ...

class Qwen3OmniMoeTalkerTextConfig(Qwen3MoeConfig):
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
        attention_bias=...,
        sliding_window=...,
        attention_dropout=...,
        decoder_sparse_step=...,
        moe_intermediate_size=...,
        num_experts_per_tok=...,
        num_experts=...,
        norm_topk_prob=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        mlp_only_layers=...,
        **kwargs,
    ) -> None: ...

class Qwen3OmniMoeTalkerConfig(PretrainedConfig):
    sub_configs = ...
    def __init__(
        self,
        code_predictor_config=...,
        text_config=...,
        num_code_groups=...,
        thinker_hidden_size=...,
        codec_eos_token_id=...,
        accept_hidden_layer=...,
        codec_nothink_id=...,
        codec_think_bos_id=...,
        codec_think_eos_id=...,
        codec_pad_id=...,
        codec_bos_id=...,
        audio_token_id=...,
        image_token_id=...,
        video_token_id=...,
        vision_start_token_id=...,
        position_id_per_seconds=...,
        audio_start_token_id=...,
        speaker_id=...,
        **kwargs,
    ) -> None: ...

class Qwen3OmniMoeCode2WavConfig(PretrainedConfig):
    def __init__(
        self,
        codebook_size=...,
        hidden_size=...,
        max_position_embeddings=...,
        rope_theta=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        attention_bias=...,
        sliding_window=...,
        intermediate_size=...,
        hidden_act=...,
        layer_scale_initial_scale=...,
        rms_norm_eps=...,
        num_hidden_layers=...,
        num_quantizers=...,
        upsample_rates=...,
        upsampling_ratios=...,
        decoder_dim=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...
    @property
    def layer_types(self):  # -> list[str]:
        ...

class Qwen3OmniMoeConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        thinker_config=...,
        talker_config=...,
        code2wav_config=...,
        enable_audio_output=...,
        im_start_token_id=...,
        im_end_token_id=...,
        tts_pad_token_id=...,
        tts_bos_token_id=...,
        tts_eos_token_id=...,
        system_token_id=...,
        user_token_id=...,
        assistant_token_id=...,
        **kwargs,
    ) -> None: ...
    def get_text_config(self, decoder=...) -> PretrainedConfig: ...

class Qwen3OmniMoePreTrainedModel(Qwen2_5OmniPreTrainedModel): ...

class Qwen3OmniMoePreTrainedModelForConditionalGeneration(
    Qwen2_5OmniPreTrainedModelForConditionalGeneration
):
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

class Qwen3OmniMoeAudioAttention(Qwen2_5OmniAudioAttention):
    def __init__(self, config) -> None: ...

class Qwen3OmniMoeAudioEncoder(Qwen2_5OmniAudioEncoder):
    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None: ...
    def forward(
        self, input_features, feature_lens=..., aftercnn_lens=...
    ):  # -> BaseModelOutput:
        ...

class Qwen3OmniMoeVisionAttention(Qwen3VLMoeVisionAttention):
    def __init__(self, config: Qwen3OmniMoeVisionEncoderConfig) -> None: ...

class Qwen3OmniMoeVisionPatchMerger(nn.Module):
    def __init__(
        self, config: Qwen3OmniMoeVisionEncoderConfig, use_postshuffle_norm=...
    ) -> None: ...
    def forward(self, hidden: torch.Tensor) -> torch.Tensor: ...

class Qwen3OmniMoeVisionEncoder(Qwen3VLMoeVisionModel):
    config: Qwen3OmniMoeVisionEncoderConfig
    _no_split_modules = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @property
    def deepstack_merger_list(self):  # -> ModuleList:
        ...

class Qwen3OmniMoeThinkerTextRotaryEmbedding(Qwen3VLMoeTextRotaryEmbedding): ...
class Qwen3OmniMoeThinkerTextSparseMoeBlock(Qwen3MoeSparseMoeBlock): ...

class Qwen3OmniMoeThinkerTextAttention(Qwen3MoeAttention):
    def __init__(self, config, layer_idx) -> None: ...

class Qwen3OmniMoeThinkerTextDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx) -> None: ...

class Qwen3OmniMoeThinkerTextPreTrainedModel(Qwen3MoePreTrainedModel):
    config_class = Qwen3OmniMoeTextConfig
    config = ...

class Qwen3OmniMoeThinkerTextModel(Qwen3VLMoeTextModel):
    config_class = Qwen3OmniMoeTextConfig
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTextConfig) -> None: ...

@dataclass
class Qwen3OmniMoeThinkerCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    rope_deltas: torch.LongTensor | None = ...

class Qwen3OmniMoeThinkerForConditionalGeneration(
    Qwen2_5OmniThinkerForConditionalGeneration
):
    _no_split_modules = ...
    _can_record_outputs = ...
    def __init__(self, config) -> None: ...
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = ...,
        audio_feature_lengths: torch.LongTensor | None = ...,
    ):  # -> Any:
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

class Qwen3OmniMoeTalkerResizeMLP(nn.Module):
    def __init__(self, config: Qwen3OmniMoeTalkerConfig) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

@dataclass
class Qwen3OmniMoeTalkerCodePredictorOutputWithPast(CausalLMOutputWithPast):
    generation_steps: int | None = ...

class Qwen3OmniMoeTalkerCodePredictorAttention(Qwen3Attention): ...

class Qwen3OmniMoeTalkerCodePredictorDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx) -> None: ...

class Qwen3OmniMoeTalkerCodePredictorModel(Qwen3Model):
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = ...
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig) -> None: ...
    def get_input_embeddings(self):  # -> ModuleList:
        ...
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

class Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(Qwen3ForCausalLM):
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = ...
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig) -> None: ...
    def get_input_embeddings(self):  # -> ModuleList:
        ...
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
    ):  # -> Qwen3OmniMoeTalkerCodePredictorOutputWithPast:
        ...

@dataclass
class Qwen3OmniMoeTalkerOutputWithPast(MoeCausalLMOutputWithPast):
    generation_step: int | None = ...

class Qwen3OmniMoeTalkerRotaryEmbedding(Qwen3OmniMoeThinkerTextRotaryEmbedding): ...
class Qwen3OmniMoeTalkerTextMLP(Qwen3MoeMLP): ...
class Qwen3OmniMoeTalkerTextSparseMoeBlock(Qwen2MoeSparseMoeBlock): ...

class Qwen3OmniMoeTalkerDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx) -> None: ...

class Qwen3OmniMoeTalkerModel(Qwen3VLMoeTextModel):
    config_class = Qwen3OmniMoeTalkerTextConfig
    base_model_prefix = ...
    _no_split_modules = ...
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTalkerTextConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...

class Qwen3OmniMoeTalkerForConditionalGeneration(Qwen3MoeForCausalLM):
    config_class = Qwen3OmniMoeTalkerConfig
    base_model_prefix = ...
    _no_split_modules = ...
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeTalkerConfig) -> None: ...
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
    ):  # -> Qwen3OmniMoeTalkerOutputWithPast:
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

class Qwen3OmniMoeCode2WavRotatoryEmbedding(Qwen3RotaryEmbedding): ...

class Qwen3OmniMoeCode2WavAttention(Qwen3Attention):
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx) -> None: ...

class Qwen3OmniMoeCode2WavMlp(Qwen3MLP): ...
class Qwen3OmniMoeCode2WavRMSNorm(Qwen3RMSNorm): ...
class Qwen3OmniMoeCode2WavLayerScale(MimiLayerScale): ...

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

class Qwen3OmniMoeCode2WavTransformerModel(Qwen3Model):
    _can_record_outputs = ...
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig) -> None: ...
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
    ):  # -> BaseModelOutputWithPast:
        ...

class SnakeBeta(SnakeBeta): ...

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

class Qwen3OmniMoeProcessorKwargs(Qwen2_5OmniProcessorKwargs):
    _defaults = ...

class Qwen3OmniMoeProcessor(Qwen2_5OmniProcessor, ProcessorMixin):
    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):  # -> list[Any]:
        ...
    def __call__(
        self,
        text: TextInput = ...,
        images: ImageInput = ...,
        videos: VideoInput = ...,
        audio: AudioInput = ...,
        **kwargs,
    ):  # -> BatchFeature:
        ...
    def apply_chat_template(
        self, conversations, chat_template=..., **kwargs
    ):  # -> str:
        ...

__all__ = [
    "Qwen3OmniMoeCode2Wav",
    "Qwen3OmniMoeCode2WavDecoderBlock",
    "Qwen3OmniMoeCode2WavTransformerModel",
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen3OmniMoePreTrainedModel",
    ...,
    "Qwen3OmniMoeProcessor",
    "Qwen3OmniMoeTalkerCodePredictorModel",
    ...,
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeTalkerForConditionalGeneration",
    "Qwen3OmniMoeTalkerModel",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
    "Qwen3OmniMoeThinkerTextModel",
    "Qwen3OmniMoeThinkerTextPreTrainedModel",
]
