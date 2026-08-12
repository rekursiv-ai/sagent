from ...configuration_utils import PretrainedConfig

"""ColPali model configuration"""
logger = ...

class ColPaliConfig(PretrainedConfig):
    def __init__(
        self, vlm_config=..., text_config=..., embedding_dim: int = ..., **kwargs
    ) -> None: ...

__all__ = ["ColPaliConfig"]
