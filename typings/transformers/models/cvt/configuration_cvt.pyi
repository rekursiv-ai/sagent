from ...configuration_utils import PretrainedConfig

"""CvT model configuration"""
logger = ...

class CvtConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        patch_sizes=...,
        patch_stride=...,
        patch_padding=...,
        embed_dim=...,
        num_heads=...,
        depth=...,
        mlp_ratio=...,
        attention_drop_rate=...,
        drop_rate=...,
        drop_path_rate=...,
        qkv_bias=...,
        cls_token=...,
        qkv_projection_method=...,
        kernel_qkv=...,
        padding_kv=...,
        stride_kv=...,
        padding_q=...,
        stride_q=...,
        initializer_range=...,
        layer_norm_eps=...,
        **kwargs,
    ) -> None: ...

__all__ = ["CvtConfig"]
