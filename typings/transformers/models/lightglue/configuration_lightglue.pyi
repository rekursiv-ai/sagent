from ..superpoint import SuperPointConfig
from ...configuration_utils import PretrainedConfig

class LightGlueConfig(PretrainedConfig):
    def __init__(
        self,
        keypoint_detector_config: SuperPointConfig = ...,
        descriptor_dim: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        num_key_value_heads=...,
        depth_confidence: float = ...,
        width_confidence: float = ...,
        filter_threshold: float = ...,
        initializer_range: float = ...,
        hidden_act: str = ...,
        attention_dropout=...,
        attention_bias=...,
        trust_remote_code: bool = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["LightGlueConfig"]
