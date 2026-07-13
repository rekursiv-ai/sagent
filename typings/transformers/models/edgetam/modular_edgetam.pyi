from transformers.models.sam2.configuration_sam2 import (
    Sam2Config,
    Sam2MaskDecoderConfig,
    Sam2PromptEncoderConfig,
)
from transformers.models.sam2.modeling_sam2 import (
    Sam2Attention,
    Sam2FeedForward,
    Sam2LayerNorm,
    Sam2Model,
    Sam2PreTrainedModel,
    Sam2TwoWayAttentionBlock,
    Sam2VisionEncoderOutput,
    Sam2VisionModel,
)
from transformers.utils.generic import TransformersKwargs, check_model_inputs

import torch

from ...configuration_utils import PretrainedConfig
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""PyTorch SAM 2 model."""

class EdgeTamVisionConfig(PretrainedConfig):
    base_config_key = ...
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        backbone_config=...,
        backbone_channel_list=...,
        backbone_feature_sizes=...,
        fpn_hidden_size=...,
        fpn_kernel_size=...,
        fpn_stride=...,
        fpn_padding=...,
        fpn_top_down_levels=...,
        num_feature_levels=...,
        hidden_act=...,
        layer_norm_eps=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class EdgeTamPromptEncoderConfig(Sam2PromptEncoderConfig): ...
class EdgeTamMaskDecoderConfig(Sam2MaskDecoderConfig): ...
class EdgeTamConfig(Sam2Config): ...
class EdgeTamLayerNorm(Sam2LayerNorm): ...
class EdgeTamVisionEncoderOutput(Sam2VisionEncoderOutput): ...
class EdgeTamAttention(Sam2Attention): ...
class EdgeTamTwoWayAttentionBlock(Sam2TwoWayAttentionBlock): ...
class EdgeTamFeedForward(Sam2FeedForward): ...

@auto_docstring
class EdgeTamPreTrainedModel(Sam2PreTrainedModel): ...

@auto_docstring(custom_intro=...)
class EdgeTamVisionModel(Sam2VisionModel):
    config_class = EdgeTamVisionConfig
    main_input_name = ...
    _can_record_outputs = ...
    def get_input_embeddings(self): ...
    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | EdgeTamVisionEncoderOutput: ...

class EdgeTamModel(Sam2Model):
    _keys_to_ignore_on_load_unexpected = ...
    def get_input_embeddings(self): ...

__all__ = [
    "EdgeTamConfig",
    "EdgeTamMaskDecoderConfig",
    "EdgeTamModel",
    "EdgeTamPreTrainedModel",
    "EdgeTamPromptEncoderConfig",
    "EdgeTamVisionConfig",
    "EdgeTamVisionModel",
]
