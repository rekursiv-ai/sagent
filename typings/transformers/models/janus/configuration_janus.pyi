from ...configuration_utils import PretrainedConfig

logger = ...

class JanusVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        patch_size=...,
        image_size=...,
        attention_dropout=...,
        layer_norm_eps=...,
        hidden_act=...,
        mlp_ratio=...,
        attention_bias=...,
        hidden_dropout_rate=...,
        projection_dim=...,
        projection_dropout=...,
        use_qk_norm=...,
        initializer_range=...,
        depth=...,
        num_image_tokens=...,
        **kwargs,
    ) -> None: ...

class JanusVQVAEConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        embed_dim: int = ...,
        num_embeddings: int = ...,
        double_latent: bool = ...,
        latent_channels: int = ...,
        num_patches: int = ...,
        in_channels: int = ...,
        out_channels: int = ...,
        base_channels: int = ...,
        channel_multiplier: list[int] = ...,
        num_res_blocks: int = ...,
        dropout: float = ...,
        initializer_range=...,
        projection_dim=...,
        num_hidden_layers=...,
        hidden_act=...,
        image_token_embed_dim=...,
        **kwargs,
    ) -> None: ...

class JanusConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        vq_config=...,
        image_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["JanusConfig", "JanusVQVAEConfig", "JanusVisionConfig"]
