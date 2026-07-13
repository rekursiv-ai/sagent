from ...configuration_utils import PretrainedConfig

"""CLAP model configuration"""
logger = ...

class ClapTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_factor=...,
        layer_norm_eps=...,
        projection_dim=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        position_embedding_type=...,
        use_cache=...,
        projection_hidden_act=...,
        **kwargs,
    ) -> None: ...

class ClapAudioConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        window_size=...,
        num_mel_bins=...,
        spec_size=...,
        hidden_act=...,
        patch_size=...,
        patch_stride=...,
        num_classes=...,
        hidden_size=...,
        projection_dim=...,
        depths=...,
        num_attention_heads=...,
        enable_fusion=...,
        hidden_dropout_prob=...,
        fusion_type=...,
        patch_embed_input_channels=...,
        flatten_patch_embeds=...,
        patch_embeds_hidden_size=...,
        enable_patch_layer_norm=...,
        drop_path_rate=...,
        attention_probs_dropout_prob=...,
        qkv_bias=...,
        mlp_ratio=...,
        aff_block_r=...,
        num_hidden_layers=...,
        projection_hidden_act=...,
        layer_norm_eps=...,
        initializer_factor=...,
        **kwargs,
    ) -> None: ...

class ClapConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        audio_config=...,
        logit_scale_init_value=...,
        projection_dim=...,
        projection_hidden_act=...,
        initializer_factor=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ClapAudioConfig", "ClapConfig", "ClapTextConfig"]
