from collections.abc import Callable

from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""Pvt V2 model configuration"""
logger = ...

class PvtV2Config(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size: int | tuple[int, int] = ...,
        num_channels: int = ...,
        num_encoder_blocks: int = ...,
        depths: list[int] = ...,
        sr_ratios: list[int] = ...,
        hidden_sizes: list[int] = ...,
        patch_sizes: list[int] = ...,
        strides: list[int] = ...,
        num_attention_heads: list[int] = ...,
        mlp_ratios: list[int] = ...,
        hidden_act: str | Callable = ...,
        hidden_dropout_prob: float = ...,
        attention_probs_dropout_prob: float = ...,
        initializer_range: float = ...,
        drop_path_rate: float = ...,
        layer_norm_eps: float = ...,
        qkv_bias: bool = ...,
        linear_attention: bool = ...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["PvtV2Config"]
