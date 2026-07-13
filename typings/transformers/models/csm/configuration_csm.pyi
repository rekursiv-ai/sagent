from ...configuration_utils import PretrainedConfig

logger = ...

class CsmDepthDecoderConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        num_codebooks=...,
        backbone_hidden_size=...,
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
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        head_dim=...,
        **kwargs,
    ) -> None: ...

class CsmConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    keys_to_ignore_at_inference = ...
    sub_configs = ...
    def __init__(
        self,
        num_codebooks=...,
        vocab_size=...,
        text_vocab_size=...,
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
        pad_token_id=...,
        codebook_pad_token_id=...,
        codebook_eos_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        audio_token_id=...,
        audio_eos_token_id=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        head_dim=...,
        tie_codebooks_embeddings=...,
        depth_decoder_config=...,
        codec_config=...,
        **kwargs,
    ) -> None: ...

__all__ = ["CsmConfig", "CsmDepthDecoderConfig"]
