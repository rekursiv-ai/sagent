from ...configuration_utils import PretrainedConfig

"""chameleon model configuration"""
logger = ...

class ChameleonVQVAEConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        embed_dim: int = ...,
        num_embeddings: int = ...,
        double_latent: bool = ...,
        latent_channels: int = ...,
        resolution: int = ...,
        in_channels: int = ...,
        base_channels: int = ...,
        channel_multiplier: list[int] = ...,
        num_res_blocks: int = ...,
        attn_resolutions: list[int] | None = ...,
        dropout: float = ...,
        attn_type: str = ...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class ChameleonConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
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
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        model_parallel_size=...,
        swin_norm=...,
        vq_config=...,
        vocabulary_map=...,
        mlp_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ChameleonConfig", "ChameleonVQVAEConfig"]
