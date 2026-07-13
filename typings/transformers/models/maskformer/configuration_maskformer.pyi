from ...configuration_utils import PretrainedConfig

"""MaskFormer model configuration"""
logger = ...

class MaskFormerConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    backbones_supported = ...
    decoders_supported = ...
    def __init__(
        self,
        fpn_feature_size: int = ...,
        mask_feature_size: int = ...,
        no_object_weight: float = ...,
        use_auxiliary_loss: bool = ...,
        backbone_config: dict | None = ...,
        decoder_config: dict | None = ...,
        init_std: float = ...,
        init_xavier_std: float = ...,
        dice_weight: float = ...,
        cross_entropy_weight: float = ...,
        mask_weight: float = ...,
        output_auxiliary_logits: bool | None = ...,
        backbone: str | None = ...,
        use_pretrained_backbone: bool = ...,
        use_timm_backbone: bool = ...,
        backbone_kwargs: dict | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(self):  # -> dict[Any, Any]:
        ...
    @classmethod
    def from_backbone_and_decoder_configs(
        cls,
        backbone_config: PretrainedConfig,
        decoder_config: PretrainedConfig,
        **kwargs,
    ):  # -> Self:
        ...

__all__ = ["MaskFormerConfig"]
