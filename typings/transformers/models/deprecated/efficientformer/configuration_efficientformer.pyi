from ....configuration_utils import PretrainedConfig

"""EfficientFormer model configuration"""
logger = ...

class EfficientFormerConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        depths: list[int] = ...,
        hidden_sizes: list[int] = ...,
        downsamples: list[bool] = ...,
        dim: int = ...,
        key_dim: int = ...,
        attention_ratio: int = ...,
        resolution: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        mlp_expansion_ratio: int = ...,
        hidden_dropout_prob: float = ...,
        patch_size: int = ...,
        num_channels: int = ...,
        pool_size: int = ...,
        downsample_patch_size: int = ...,
        downsample_stride: int = ...,
        downsample_pad: int = ...,
        drop_path_rate: float = ...,
        num_meta3d_blocks: int = ...,
        distillation: bool = ...,
        use_layer_scale: bool = ...,
        layer_scale_init_value: float = ...,
        hidden_act: str = ...,
        initializer_range: float = ...,
        layer_norm_eps: float = ...,
        image_size: int = ...,
        batch_norm_eps: float = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["EfficientFormerConfig"]
