from ...configuration_utils import PretrainedConfig

"""PaliGemmamodel configuration"""
logger = ...

class PaliGemmaConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_index=...,
        vocab_size=...,
        projection_dim=...,
        hidden_size=...,
        **kwargs,
    ) -> None: ...

__all__ = ["PaliGemmaConfig"]
