from ...configuration_utils import PretrainedConfig

"""VitPose model configuration"""
logger = ...

class VitPoseConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        backbone_config: PretrainedConfig | None = ...,
        backbone: str | None = ...,
        use_pretrained_backbone: bool = ...,
        use_timm_backbone: bool = ...,
        backbone_kwargs: dict | None = ...,
        initializer_range: float = ...,
        scale_factor: int = ...,
        use_simple_decoder: bool = ...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(self):  # -> dict[str, type[PretrainedConfig] | type[None]]:
        ...

__all__ = ["VitPoseConfig"]
