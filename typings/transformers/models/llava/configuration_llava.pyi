from ...configuration_utils import PretrainedConfig

"""Llava model configuration"""
logger = ...

class LlavaConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_index=...,
        projector_hidden_act=...,
        vision_feature_select_strategy=...,
        vision_feature_layer=...,
        image_seq_length=...,
        multimodal_projector_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["LlavaConfig"]
