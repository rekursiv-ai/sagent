from dataclasses import dataclass

from torch import nn

import torch

from .configuration_univnet import UnivNetConfig
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch UnivNetModel model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class UnivNetModelOutput(ModelOutput):
    waveforms: torch.FloatTensor | None = ...
    waveform_lengths: torch.FloatTensor | None = ...

class UnivNetKernelPredictorResidualBlock(nn.Module):
    def __init__(self, config: UnivNetConfig) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor):  # -> Tensor:
        ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...

class UnivNetKernelPredictor(nn.Module):
    def __init__(
        self, config: UnivNetConfig, conv_kernel_size: int = ..., conv_layers: int = ...
    ) -> None: ...
    def forward(self, spectrogram: torch.FloatTensor):  # -> tuple[Any, Any]:
        ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...

class UnivNetLvcResidualBlock(nn.Module):
    def __init__(
        self, config: UnivNetConfig, kernel_size: int, dilation: int
    ) -> None: ...
    def forward(self, hidden_states, kernel, bias, hop_size=...): ...
    def location_variable_convolution(
        self,
        hidden_states: torch.FloatTensor,
        kernel: torch.FloatTensor,
        bias: torch.FloatTensor,
        dilation: int = ...,
        hop_size: int = ...,
    ):  # -> Tensor:
        ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...

class UnivNetLvcBlock(nn.Module):
    def __init__(
        self, config: UnivNetConfig, layer_id: int, lvc_hop_size: int = ...
    ) -> None: ...
    def forward(
        self, hidden_states: torch.FloatTensor, spectrogram: torch.FloatTensor
    ):  # -> FloatTensor:
        ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...

@auto_docstring
class UnivNetModel(PreTrainedModel):
    config: UnivNetConfig
    main_input_name = ...
    def __init__(self, config: UnivNetConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor,
        noise_sequence: torch.FloatTensor | None = ...,
        padding_mask: torch.FloatTensor | None = ...,
        generator: torch.Generator | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | UnivNetModelOutput: ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...

__all__ = ["UnivNetModel"]
