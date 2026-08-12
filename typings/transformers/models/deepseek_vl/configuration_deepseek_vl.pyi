from ..auto import AutoConfig
from ...configuration_utils import PretrainedConfig

logger = ...

class DeepseekVLConfig(PretrainedConfig):
    def __init__(
        self,
        text_config: AutoConfig | None = ...,
        vision_config: AutoConfig | None = ...,
        image_token_id: int = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["DeepseekVLConfig"]
