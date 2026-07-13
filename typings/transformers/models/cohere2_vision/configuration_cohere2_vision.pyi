from ...configuration_utils import PretrainedConfig

class Cohere2VisionConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        downsample_factor=...,
        image_token_id=...,
        alignment_intermediate_size=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Cohere2VisionConfig"]
