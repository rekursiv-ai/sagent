from ...configuration_utils import PretrainedConfig

class Mistral3Config(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    is_composition = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_index=...,
        projector_hidden_act=...,
        vision_feature_layer=...,
        multimodal_projector_bias=...,
        spatial_merge_size=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Mistral3Config"]
