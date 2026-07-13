from ...configuration_utils import PretrainedConfig

"""Idefics model configuration"""
logger = ...

class IdeficsVisionConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        embed_dim=...,
        image_size=...,
        intermediate_size=...,
        patch_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        initializer_factor=...,
        **kwargs,
    ) -> None: ...

class IdeficsPerceiverConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        use_resampler=...,
        resampler_n_latents=...,
        resampler_depth=...,
        resampler_n_heads=...,
        resampler_head_dim=...,
        qk_layer_norms_perceiver=...,
        **kwargs,
    ) -> None: ...

class IdeficsConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        vocab_size=...,
        additional_vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        dropout=...,
        hidden_act=...,
        initializer_range=...,
        alpha_initializer=...,
        alphas_initializer_range=...,
        alpha_type=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        cross_layer_interval=...,
        qk_layer_norms=...,
        freeze_text_layers=...,
        freeze_text_module_exceptions=...,
        freeze_lm_head=...,
        freeze_vision_layers=...,
        freeze_vision_module_exceptions=...,
        use_resampler=...,
        vision_config=...,
        perceiver_config=...,
        **kwargs,
    ) -> None: ...

__all__ = ["IdeficsConfig"]
