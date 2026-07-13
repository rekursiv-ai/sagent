from typing import Any

from torch import nn

import torch

from ..rt_detr.modeling_rt_detr import (
    RTDetrConvNormLayer,
    RTDetrDecoder,
    RTDetrDecoderLayer,
    RTDetrDecoderOutput,
    RTDetrEncoder,
    RTDetrForObjectDetection,
    RTDetrHybridEncoder,
    RTDetrMLPPredictionHead,
    RTDetrModel,
    RTDetrPreTrainedModel,
    RTDetrRepVggBlock,
)
from ...configuration_utils import PretrainedConfig

logger = ...

class DFineConfig(PretrainedConfig):
    model_type = ...
    layer_types = ...
    attribute_map = ...
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
        weight_loss_fgl=...,
        weight_loss_ddf=...,
        eos_coefficient=...,
        eval_idx=...,
        layer_scale=...,
        max_num_bins=...,
        reg_scale=...,
        depth_mult=...,
        top_prob_values=...,
        lqe_hidden_dim=...,
        lqe_layers=...,
        decoder_offset_scale=...,
        decoder_method=...,
        up=...,
        **kwargs,
    ) -> None: ...
    @property
    def num_attention_heads(self) -> int: ...
    @property
    def hidden_size(self) -> int: ...
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

class DFineMultiscaleDeformableAttention(nn.Module):
    def __init__(self, config: DFineConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        reference_points=...,
        encoder_hidden_states=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class DFineGate(nn.Module):
    def __init__(self, d_model: int) -> None: ...
    def forward(
        self, second_residual: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class DFineDecoderLayer(RTDetrDecoderLayer):
    def __init__(self, config: DFineConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = ...,
        reference_points=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, Any, Any]: ...

class DFinePreTrainedModel(RTDetrPreTrainedModel): ...

class DFineIntegral(nn.Module):
    def __init__(self, config: DFineConfig) -> None: ...
    def forward(
        self, pred_corners: torch.Tensor, project: torch.Tensor
    ) -> torch.Tensor: ...

class DFineDecoderOutput(RTDetrDecoderOutput): ...

class DFineDecoder(RTDetrDecoder):
    def __init__(self, config: DFineConfig) -> None: ...
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spatial_shapes,
        level_start_index=...,
        spatial_shapes_list=...,
        output_hidden_states=...,
        encoder_attention_mask=...,
        memory_mask=...,
        output_attentions=...,
        return_dict=...,
    ) -> DFineDecoderOutput: ...

class DFineModel(RTDetrModel):
    def __init__(self, config: DFineConfig) -> None: ...

class DFineForObjectDetection(RTDetrForObjectDetection, DFinePreTrainedModel):
    def __init__(self, config: DFineConfig) -> None: ...
    def forward(**super_kwargs):  # -> None:
        ...

def weighting_function(
    max_num_bins: int, up: torch.Tensor, reg_scale: int
) -> torch.Tensor: ...

class DFineMLPPredictionHead(RTDetrMLPPredictionHead): ...

def distance2bbox(points, distance: torch.Tensor, reg_scale: float) -> torch.Tensor: ...

class DFineMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: str = ...,
    ) -> None: ...
    def forward(self, stat_features: torch.Tensor) -> torch.Tensor: ...

class DFineLQE(nn.Module):
    def __init__(self, config: DFineConfig) -> None: ...
    def forward(
        self, scores: torch.Tensor, pred_corners: torch.Tensor
    ) -> torch.Tensor: ...

class DFineConvNormLayer(RTDetrConvNormLayer):
    def __init__(
        self,
        config: DFineConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = ...,
        padding: int | None = ...,
        activation: str | None = ...,
    ) -> None: ...

class DFineRepVggBlock(RTDetrRepVggBlock):
    def __init__(
        self, config: DFineConfig, in_channels: int, out_channels: int
    ) -> None: ...

class DFineCSPRepLayer(nn.Module):
    def __init__(
        self,
        config: DFineConfig,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        expansion: float = ...,
    ) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...

class DFineRepNCSPELAN4(nn.Module):
    def __init__(
        self, config: DFineConfig, act: str = ..., numb_blocks: int = ...
    ) -> None: ...
    def forward(self, input_features: torch.Tensor) -> torch.Tensor: ...

class DFineSCDown(nn.Module):
    def __init__(self, config: DFineConfig, kernel_size: int, stride: int) -> None: ...
    def forward(self, input_features: torch.Tensor) -> torch.Tensor: ...

class DFineEncoder(RTDetrEncoder): ...

class DFineHybridEncoder(RTDetrHybridEncoder):
    def __init__(self, config: DFineConfig) -> None: ...

__all__ = [
    "DFineConfig",
    "DFineForObjectDetection",
    "DFineModel",
    "DFinePreTrainedModel",
]
