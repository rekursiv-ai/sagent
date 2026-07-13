from dataclasses import dataclass

from torch import nn

import torch

from .configuration_mgp_str import MgpstrConfig
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch MGP-STR model."""
logger = ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class MgpstrDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

@dataclass
@auto_docstring(custom_intro=...)
class MgpstrModelOutput(ModelOutput):
    logits: tuple[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    a3_attentions: tuple[torch.FloatTensor] | None = ...

class MgpstrEmbeddings(nn.Module):
    def __init__(self, config: MgpstrConfig) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class MgpstrMlp(nn.Module):
    def __init__(self, config: MgpstrConfig, hidden_features) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class MgpstrAttention(nn.Module):
    def __init__(self, config: MgpstrConfig) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class MgpstrLayer(nn.Module):
    def __init__(self, config: MgpstrConfig, drop_path=...) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class MgpstrEncoder(nn.Module):
    def __init__(self, config: MgpstrConfig) -> None: ...
    def forward(
        self,
        hidden_states,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

class MgpstrA3Module(nn.Module):
    def __init__(self, config: MgpstrConfig) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Tensor]:
        ...

@auto_docstring
class MgpstrPreTrainedModel(PreTrainedModel):
    config: MgpstrConfig
    base_model_prefix = ...
    _no_split_modules = ...

@auto_docstring
class MgpstrModel(MgpstrPreTrainedModel):
    def __init__(self, config: MgpstrConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class MgpstrForSceneTextRecognition(MgpstrPreTrainedModel):
    config: MgpstrConfig
    main_input_name = ...
    def __init__(self, config: MgpstrConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_a3_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | MgpstrModelOutput: ...

__all__ = ["MgpstrForSceneTextRecognition", "MgpstrModel", "MgpstrPreTrainedModel"]
