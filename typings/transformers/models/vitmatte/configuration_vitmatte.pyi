from ...configuration_utils import PretrainedConfig

"""VitMatte model configuration"""
logger = ...

class VitMatteConfig(PretrainedConfig):
    def __init__(
        self,
        backbone_config: PretrainedConfig | None = ...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        backbone_kwargs=...,
        hidden_size: int = ...,
        batch_norm_eps: float = ...,
        initializer_range: float = ...,
        convstream_hidden_sizes: list[int] = ...,
        fusion_hidden_sizes: list[int] = ...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(self):  # -> dict[str, type[PretrainedConfig] | type[None]]:
        ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

__all__ = ["VitMatteConfig"]
