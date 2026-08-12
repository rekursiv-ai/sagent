from ...configuration_utils import PretrainedConfig

logger = ...

class Qwen3OmniMoeAudioEncoderConfig(PretrainedConfig):
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

class Qwen3OmniMoeVisionEncoderConfig(PretrainedConfig):
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
        out_hidden_size=...,
        num_position_embeddings=...,
        deepstack_visual_indexes=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Qwen3OmniMoeTextConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
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

class Qwen3OmniMoeThinkerConfig(PretrainedConfig):
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

class Qwen3OmniMoeTalkerCodePredictorConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
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

class Qwen3OmniMoeTalkerTextConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
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

__all__ = [
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeThinkerConfig",
]
