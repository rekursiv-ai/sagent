from ...configuration_utils import PretrainedConfig

logger = ...

class LlavaOnevisionConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_index=...,
        video_token_index=...,
        projector_hidden_act=...,
        vision_feature_select_strategy=...,
        vision_feature_layer=...,
        vision_aspect_ratio=...,
        image_grid_pinpoints=...,
        tie_word_embeddings=...,
        multimodal_projector_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["LlavaOnevisionConfig"]
