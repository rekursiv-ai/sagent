from ...configuration_utils import PretrainedConfig

"""VisionTextDualEncoder model configuration"""
logger = ...
VISION_MODEL_CONFIGS = ...

class VisionTextDualEncoderConfig(PretrainedConfig):
    def __init__(
        self, projection_dim=..., logit_scale_init_value=..., **kwargs
    ) -> None: ...
    @classmethod
    def from_vision_text_configs(
        cls, vision_config: PretrainedConfig, text_config: PretrainedConfig, **kwargs
    ):  # -> Self:
        ...

__all__ = ["VisionTextDualEncoderConfig"]
