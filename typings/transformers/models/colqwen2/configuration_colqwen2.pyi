from typing import Any

from ...configuration_utils import PretrainedConfig

logger = ...

class ColQwen2Config(PretrainedConfig):
    sub_configs: dict[str, Any] = ...
    def __init__(
        self,
        vlm_config=...,
        embedding_dim: int = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...
    def get_text_config(self, *args, **kwargs) -> PretrainedConfig: ...

__all__ = ["ColQwen2Config"]
