from ...configuration_utils import PretrainedConfig

"""PyTorch LFM2-VL model."""
logger = ...

class Lfm2VlConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_id=...,
        projector_hidden_act=...,
        projector_hidden_size=...,
        projector_bias=...,
        downsample_factor=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Lfm2VlConfig"]
