from torch import Tensor, nn

import torch

from ..rt_detr.modeling_rt_detr import (
    RTDetrDecoder,
    RTDetrDecoderLayer,
    RTDetrForObjectDetection,
    RTDetrMLPPredictionHead,
    RTDetrModel,
    RTDetrPreTrainedModel,
)
from ...configuration_utils import PretrainedConfig

logger = ...

class RTDetrV2Config(PretrainedConfig):
    layer_types = ...
    def __init__(
        self,
        initializer_range=...,
        initializer_bias_prior_prob=...,
        layer_norm_eps=...,
        batch_norm_eps=...,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        freeze_backbone_batch_norms=...,
        backbone_kwargs=...,
        encoder_hidden_dim=...,
        encoder_in_channels=...,
        feat_strides=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        dropout=...,
        activation_dropout=...,
        encode_proj_layers=...,
        positional_encoding_temperature=...,
        encoder_activation_function=...,
        activation_function=...,
        eval_size=...,
        normalize_before=...,
        hidden_expansion=...,
        d_model=...,
        num_queries=...,
        decoder_in_channels=...,
        decoder_ffn_dim=...,
        num_feature_levels=...,
        decoder_n_points=...,
        decoder_layers=...,
        decoder_attention_heads=...,
        decoder_activation_function=...,
        attention_dropout=...,
        num_denoising=...,
        label_noise_ratio=...,
        box_noise_scale=...,
        learn_initial_query=...,
        anchor_image_size=...,
        with_box_refine=...,
        is_encoder_decoder=...,
        matcher_alpha=...,
        matcher_gamma=...,
        matcher_class_cost=...,
        matcher_bbox_cost=...,
        matcher_giou_cost=...,
        use_focal_loss=...,
        auxiliary_loss=...,
        focal_loss_alpha=...,
        focal_loss_gamma=...,
        weight_loss_vfl=...,
        weight_loss_bbox=...,
        weight_loss_giou=...,
        eos_coefficient=...,
        decoder_n_levels=...,
        decoder_offset_scale=...,
        decoder_method=...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[PretrainedConfig] | type[Any] | type[None]]:
        ...
    @classmethod
    def from_backbone_configs(
        cls, backbone_config: PretrainedConfig, **kwargs
    ):  # -> Self:
        ...

def multi_scale_deformable_attention_v2(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    num_points_list: list[int],
    method=...,
) -> Tensor: ...

class RTDetrV2MultiscaleDeformableAttention(nn.Module):
    def __init__(self, config: RTDetrV2Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        position_embeddings: torch.Tensor | None = ...,
        reference_points=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
        level_start_index=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Tensor]:
        ...

class RTDetrV2DecoderLayer(RTDetrDecoderLayer):
    def __init__(self, config: RTDetrV2Config) -> None: ...

class RTDetrV2PreTrainedModel(RTDetrPreTrainedModel): ...

class RTDetrV2Decoder(RTDetrDecoder):
    def __init__(self, config: RTDetrV2Config) -> None: ...

class RTDetrV2Model(RTDetrModel):
    def __init__(self, config: RTDetrV2Config) -> None: ...

class RTDetrV2MLPPredictionHead(RTDetrMLPPredictionHead): ...

class RTDetrV2ForObjectDetection(RTDetrForObjectDetection, RTDetrV2PreTrainedModel):
    def __init__(self, config: RTDetrV2Config) -> None: ...

__all__ = [
    "RTDetrV2Config",
    "RTDetrV2ForObjectDetection",
    "RTDetrV2Model",
    "RTDetrV2PreTrainedModel",
]
