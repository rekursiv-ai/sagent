from ...configuration_utils import PretrainedConfig

"""BridgeTower model configuration"""
logger = ...

class BridgeTowerVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_channels=...,
        patch_size=...,
        image_size=...,
        initializer_factor=...,
        layer_norm_eps=...,
        stop_gradient=...,
        share_layernorm=...,
        remove_last_layer=...,
        **kwargs,
    ) -> None: ...

class BridgeTowerTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        initializer_factor=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        layer_norm_eps=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        position_embedding_type=...,
        use_cache=...,
        **kwargs,
    ) -> None: ...

class BridgeTowerConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        share_cross_modal_transformer_layers=...,
        hidden_act=...,
        hidden_size=...,
        initializer_factor=...,
        layer_norm_eps=...,
        share_link_tower_layers=...,
        link_tower_type=...,
        num_attention_heads=...,
        num_hidden_layers=...,
        tie_word_embeddings=...,
        init_layernorm_from_vision_encoder=...,
        text_config=...,
        vision_config=...,
        **kwargs,
    ) -> None: ...

__all__ = ["BridgeTowerConfig", "BridgeTowerTextConfig", "BridgeTowerVisionConfig"]
