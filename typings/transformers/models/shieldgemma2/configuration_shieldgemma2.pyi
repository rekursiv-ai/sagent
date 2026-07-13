from ...configuration_utils import PretrainedConfig

logger = ...

class ShieldGemma2Config(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        mm_tokens_per_image: int = ...,
        boi_token_index: int = ...,
        eoi_token_index: int = ...,
        image_token_index: int = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["ShieldGemma2Config"]
