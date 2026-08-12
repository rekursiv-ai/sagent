from ...configuration_utils import PretrainedConfig

"""DINOv3 model configuration"""
logger = ...

class DINOv3ViTConfig(PretrainedConfig):
    def __init__(
        self,
        patch_size: int = ...,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        hidden_act: str = ...,
        attention_dropout: float = ...,
        initializer_range: float = ...,
        layer_norm_eps: float = ...,
        rope_theta: float = ...,
        image_size: int = ...,
        num_channels: int = ...,
        query_bias: bool = ...,
        key_bias: bool = ...,
        value_bias: bool = ...,
        proj_bias: bool = ...,
        mlp_bias: bool = ...,
        layerscale_value: float = ...,
        drop_path_rate: float = ...,
        use_gated_mlp: bool = ...,
        num_register_tokens: int = ...,
        pos_embed_shift: float | None = ...,
        pos_embed_jitter: float | None = ...,
        pos_embed_rescale: float | None = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["DINOv3ViTConfig"]
