from ...configuration_utils import PretrainedConfig

"""VipLlava model configuration"""
logger = ...

class VipLlavaConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_index=...,
        projector_hidden_act=...,
        projector_layernorm_eps=...,
        vision_feature_layers=...,
        image_seq_length=...,
        **kwargs,
    ) -> None: ...

__all__ = ["VipLlavaConfig"]
