from ...configuration_utils import PretrainedConfig

"""UperNet model configuration"""
logger = ...

class UperNetConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        backbone_kwargs=...,
        hidden_size=...,
        initializer_range=...,
        pool_scales=...,
        use_auxiliary_head=...,
        auxiliary_loss_weight=...,
        auxiliary_in_channels=...,
        auxiliary_channels=...,
        auxiliary_num_convs=...,
        auxiliary_concat_input=...,
        loss_ignore_index=...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[PretrainedConfig] | type[Any] | type[None]]:
        ...

__all__ = ["UperNetConfig"]
