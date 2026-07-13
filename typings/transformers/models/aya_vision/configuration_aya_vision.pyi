from ...configuration_utils import PretrainedConfig

"""AyaVision model configuration"""
logger = ...

class AyaVisionConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        vision_feature_select_strategy=...,
        vision_feature_layer=...,
        downsample_factor=...,
        adapter_layer_norm_eps=...,
        image_token_index=...,
        **kwargs,
    ) -> None: ...

__all__ = ["AyaVisionConfig"]
