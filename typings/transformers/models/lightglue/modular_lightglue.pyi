from dataclasses import dataclass

from PIL import Image
from torch import nn

import torch

from ..clip.modeling_clip import CLIPMLP
from ..llama.modeling_llama import LlamaAttention
from ..superglue.image_processing_superglue import SuperGlueImageProcessor
from ..superpoint import SuperPointConfig
from ...configuration_utils import PretrainedConfig
from ...image_utils import ImageInput
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TensorType, auto_docstring
from ...utils.generic import can_return_tuple

logger = ...

class LightGlueConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        keypoint_detector_config: SuperPointConfig = ...,
        descriptor_dim: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        num_key_value_heads=...,
        depth_confidence: float = ...,
        width_confidence: float = ...,
        filter_threshold: float = ...,
        initializer_range: float = ...,
        hidden_act: str = ...,
        attention_dropout=...,
        attention_bias=...,
        trust_remote_code: bool = ...,
        **kwargs,
    ) -> None: ...

@dataclass
@auto_docstring(custom_intro=...)
class LightGlueKeypointMatchingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    matches: torch.FloatTensor | None = ...
    matching_scores: torch.FloatTensor | None = ...
    keypoints: torch.FloatTensor | None = ...
    prune: torch.IntTensor | None = ...
    mask: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class LightGlueImageProcessor(SuperGlueImageProcessor):
    def post_process_keypoint_matching(
        self,
        outputs: LightGlueKeypointMatchingOutput,
        target_sizes: TensorType | list[tuple],
        threshold: float = ...,
    ) -> list[dict[str, torch.Tensor]]: ...
    def visualize_keypoint_matching(
        self,
        images: ImageInput,
        keypoint_matching_output: list[dict[str, torch.Tensor]],
    ) -> list[Image.Image]: ...
    def plot_keypoint_matching(
        self,
        images: ImageInput,
        keypoint_matching_output: LightGlueKeypointMatchingOutput,
    ):  # -> None:
        ...

class LightGluePositionalEncoder(nn.Module):
    def __init__(self, config: LightGlueConfig) -> None: ...
    def forward(
        self, keypoints: torch.Tensor, output_hidden_states: bool | None = ...
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class LightGlueAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class LightGlueMLP(CLIPMLP):
    def __init__(self, config: LightGlueConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class LightGlueTransformerLayer(nn.Module):
    def __init__(self, config: LightGlueConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        descriptors: torch.Tensor,
        keypoints: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor] | None, tuple[torch.Tensor] | None
    ]: ...

def sigmoid_log_double_softmax(
    similarity: torch.Tensor, matchability0: torch.Tensor, matchability1: torch.Tensor
) -> torch.Tensor: ...

class LightGlueMatchAssignmentLayer(nn.Module):
    def __init__(self, config: LightGlueConfig) -> None: ...
    def forward(
        self, descriptors: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor: ...
    def get_matchability(self, descriptors: torch.Tensor) -> torch.Tensor: ...

class LightGlueTokenConfidenceLayer(nn.Module):
    def __init__(self, config: LightGlueConfig) -> None: ...
    def forward(self, descriptors: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class LightGluePreTrainedModel(PreTrainedModel):
    config: LightGlueConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...

def get_matches_from_scores(
    scores: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]: ...
def normalize_keypoints(
    keypoints: torch.Tensor, height: int, width: int
) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class LightGlueForKeypointMatching(LightGluePreTrainedModel):
    def __init__(self, config: LightGlueConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> tuple | LightGlueKeypointMatchingOutput: ...

__all__ = [
    "LightGlueConfig",
    "LightGlueForKeypointMatching",
    "LightGlueImageProcessor",
    "LightGluePreTrainedModel",
]
