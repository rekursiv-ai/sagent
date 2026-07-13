from torch import nn
from transformers.models.depth_anything.configuration_depth_anything import (
    DepthAnythingConfig,
)
from transformers.models.depth_anything.modeling_depth_anything import (
    DepthAnythingDepthEstimationHead,
    DepthAnythingFeatureFusionLayer,
    DepthAnythingFeatureFusionStage,
    DepthAnythingForDepthEstimation,
    DepthAnythingNeck,
    DepthAnythingReassembleStage,
)

import torch

from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

class PromptDepthAnythingConfig(DepthAnythingConfig):
    model_type = ...

class PromptDepthAnythingLayer(nn.Module):
    def __init__(self, config: PromptDepthAnythingConfig) -> None: ...
    def forward(self, prompt_depth: torch.Tensor) -> torch.Tensor: ...

class PromptDepthAnythingFeatureFusionLayer(DepthAnythingFeatureFusionLayer):
    def __init__(self, config: PromptDepthAnythingConfig) -> None: ...
    def forward(
        self, hidden_state, residual=..., size=..., prompt_depth=...
    ):  # -> Any:
        ...

class PromptDepthAnythingFeatureFusionStage(DepthAnythingFeatureFusionStage):
    def forward(self, hidden_states, size=..., prompt_depth=...):  # -> list[Any]:
        ...

class PromptDepthAnythingDepthEstimationHead(DepthAnythingDepthEstimationHead):
    def forward(
        self, hidden_states: list[torch.Tensor], patch_height: int, patch_width: int
    ) -> torch.Tensor: ...

@auto_docstring
class PromptDepthAnythingPreTrainedModel(PreTrainedModel):
    config: PromptDepthAnythingConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

class PromptDepthAnythingReassembleLayer(nn.Module):
    def __init__(
        self, config: PromptDepthAnythingConfig, channels: int, factor: int
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class PromptDepthAnythingReassembleStage(DepthAnythingReassembleStage): ...

class PromptDepthAnythingNeck(DepthAnythingNeck):
    def forward(
        self,
        hidden_states: list[torch.Tensor],
        patch_height: int | None = ...,
        patch_width: int | None = ...,
        prompt_depth: torch.Tensor | None = ...,
    ) -> list[torch.Tensor]: ...

@auto_docstring(custom_intro=...)
class PromptDepthAnythingForDepthEstimation(DepthAnythingForDepthEstimation):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        prompt_depth: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | DepthEstimatorOutput: ...

__all__ = [
    "PromptDepthAnythingConfig",
    "PromptDepthAnythingForDepthEstimation",
    "PromptDepthAnythingPreTrainedModel",
]
