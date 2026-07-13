from ...configuration_utils import PretrainedConfig

"""PerceptionLM model configuration"""
logger = ...

class PerceptionLMConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        vision_use_cls_token=...,
        projector_pooling_ratio=...,
        image_token_id=...,
        video_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["PerceptionLMConfig"]
