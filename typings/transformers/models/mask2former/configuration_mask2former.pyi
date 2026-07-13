from ...configuration_utils import PretrainedConfig

"""Mask2Former model configuration"""
logger = ...

class Mask2FormerConfig(PretrainedConfig):
    model_type = ...
    backbones_supported = ...
    attribute_map = ...
    def __init__(
        self,
        backbone_config: dict | None = ...,
        feature_size: int = ...,
        mask_feature_size: int = ...,
        hidden_dim: int = ...,
        encoder_feedforward_dim: int = ...,
        activation_function: str = ...,
        encoder_layers: int = ...,
        decoder_layers: int = ...,
        num_attention_heads: int = ...,
        dropout: float = ...,
        dim_feedforward: int = ...,
        pre_norm: bool = ...,
        enforce_input_projection: bool = ...,
        common_stride: int = ...,
        ignore_value: int = ...,
        num_queries: int = ...,
        no_object_weight: float = ...,
        class_weight: float = ...,
        mask_weight: float = ...,
        dice_weight: float = ...,
        train_num_points: int = ...,
        oversample_ratio: float = ...,
        importance_sample_ratio: float = ...,
        init_std: float = ...,
        init_xavier_std: float = ...,
        use_auxiliary_loss: bool = ...,
        feature_strides: list[int] = ...,
        output_auxiliary_logits: bool | None = ...,
        backbone: str | None = ...,
        use_pretrained_backbone: bool = ...,
        use_timm_backbone: bool = ...,
        backbone_kwargs: dict | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(self):  # -> dict[str, type[dict[Any, Any]] | type[None]]:
        ...
    @classmethod
    def from_backbone_config(
        cls, backbone_config: PretrainedConfig, **kwargs
    ):  # -> Self:
        ...

__all__ = ["Mask2FormerConfig"]
