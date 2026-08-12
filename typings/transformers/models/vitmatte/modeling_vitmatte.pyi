from dataclasses import dataclass

from torch import nn

import torch

from .configuration_vitmatte import VitMatteConfig
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch ViTMatte model."""

@dataclass
@auto_docstring(custom_intro=...)
class ImageMattingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    alphas: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class VitMattePreTrainedModel(PreTrainedModel):
    config: VitMatteConfig
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

class VitMatteBasicConv3x3(nn.Module):
    def __init__(
        self, config, in_channels, out_channels, stride=..., padding=...
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class VitMatteConvStream(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values):  # -> dict[str, Any]:
        ...

class VitMatteFusionBlock(nn.Module):
    def __init__(self, config, in_channels, out_channels) -> None: ...
    def forward(self, features, detailed_feature_map):  # -> Any:
        ...

class VitMatteHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class VitMatteDetailCaptureModule(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, pixel_values):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class VitMatteForImageMatting(VitMattePreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ):  # -> ImageMattingOutput:
        ...

__all__ = ["VitMatteForImageMatting", "VitMattePreTrainedModel"]
