from ...configuration_utils import PretrainedConfig

logger = ...

class PromptDepthAnythingConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        backbone_kwargs=...,
        patch_size=...,
        initializer_range=...,
        reassemble_hidden_size=...,
        reassemble_factors=...,
        neck_hidden_sizes=...,
        fusion_hidden_size=...,
        head_in_index=...,
        head_hidden_size=...,
        depth_estimation_type=...,
        max_depth=...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[PretrainedConfig] | type[Any] | type[None]]:
        ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

__all__ = ["PromptDepthAnythingConfig"]
