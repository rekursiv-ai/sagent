from ...configuration_utils import PretrainedConfig

"""KOSMOS-2 model configuration"""
logger = ...

class Kosmos2TextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        embed_dim=...,
        layers=...,
        ffn_dim=...,
        attention_heads=...,
        activation_function=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        layerdrop=...,
        layer_norm_eps=...,
        init_std=...,
        scale_embedding=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

class Kosmos2VisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        initializer_factor=...,
        **kwargs,
    ) -> None: ...

class Kosmos2Config(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self, text_config=..., vision_config=..., latent_query_num=..., **kwargs
    ) -> None: ...

__all__ = ["Kosmos2Config"]
