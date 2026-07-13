from dataclasses import dataclass

from torch import nn

import torch

from .configuration_vitpose import VitPoseConfig
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple

"""PyTorch VitPose model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class VitPoseEstimatorOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    heatmaps: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring
class VitPosePreTrainedModel(PreTrainedModel):
    config: VitPoseConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

def flip_back(output_flipped, flip_pairs, target_type=...): ...

class VitPoseSimpleDecoder(nn.Module):
    def __init__(self, config: VitPoseConfig) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, flip_pairs: torch.Tensor | None = ...
    ) -> torch.Tensor: ...

class VitPoseClassicDecoder(nn.Module):
    def __init__(self, config: VitPoseConfig) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, flip_pairs: torch.Tensor | None = ...
    ):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class VitPoseForPoseEstimation(VitPosePreTrainedModel):
    def __init__(self, config: VitPoseConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        dataset_index: torch.Tensor | None = ...,
        flip_pairs: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VitPoseEstimatorOutput: ...

__all__ = ["VitPoseForPoseEstimation", "VitPosePreTrainedModel"]
