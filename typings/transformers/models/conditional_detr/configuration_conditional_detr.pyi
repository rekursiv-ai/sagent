from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""Conditional DETR model configuration"""
logger = ...

class ConditionalDetrConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        use_timm_backbone=...,
        backbone_config=...,
        num_channels=...,
        num_queries=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        encoder_layerdrop=...,
        decoder_layerdrop=...,
        is_encoder_decoder=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        init_xavier_std=...,
        auxiliary_loss=...,
        position_embedding_type=...,
        backbone=...,
        use_pretrained_backbone=...,
        backbone_kwargs=...,
        dilation=...,
        class_cost=...,
        bbox_cost=...,
        giou_cost=...,
        mask_loss_coefficient=...,
        dice_loss_coefficient=...,
        cls_loss_coefficient=...,
        bbox_loss_coefficient=...,
        giou_loss_coefficient=...,
        focal_alpha=...,
        **kwargs,
    ) -> None: ...
    @property
    def num_attention_heads(self) -> int: ...
    @property
    def hidden_size(self) -> int: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[None] | type[Any] | type[PretrainedConfig]]:
        ...

class ConditionalDetrOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...
    @property
    def default_onnx_opset(self) -> int: ...

__all__ = ["ConditionalDetrConfig", "ConditionalDetrOnnxConfig"]
