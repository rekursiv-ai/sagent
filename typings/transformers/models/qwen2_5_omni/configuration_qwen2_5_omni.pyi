from ...configuration_utils import PretrainedConfig

logger = ...

class Qwen2_5OmniVisionEncoderConfig(PretrainedConfig):
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

class Qwen2_5OmniAudioEncoderConfig(PretrainedConfig):
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
        use_sliding_window=...,
        sliding_window=...,
        max_window_layers=...,
        layer_types=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5OmniThinkerConfig(PretrainedConfig):
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
    def __init__(self, dit_config=..., bigvgan_config=..., **kwargs) -> None: ...

class Qwen2_5OmniConfig(PretrainedConfig):
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

__all__ = [
    "Qwen2_5OmniConfig",
    "Qwen2_5OmniTalkerConfig",
    "Qwen2_5OmniThinkerConfig",
    "Qwen2_5OmniToken2WavConfig",
]
