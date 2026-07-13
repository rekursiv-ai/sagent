from torch import nn

import torch

from .configuration_glpn import GLPNConfig
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch GLPN model."""
logger = ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class GLPNDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class GLPNOverlapPatchEmbeddings(nn.Module):
    def __init__(self, patch_size, stride, num_channels, hidden_size) -> None: ...
    def forward(self, pixel_values):  # -> tuple[Any, Any, Any]:
        ...

class GLPNEfficientSelfAttention(nn.Module):
    def __init__(
        self, config, hidden_size, num_attention_heads, sequence_reduction_ratio
    ) -> None: ...
    def forward(
        self, hidden_states, height, width, output_attentions=...
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class GLPNSelfOutput(nn.Module):
    def __init__(self, config, hidden_size) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

class GLPNAttention(nn.Module):
    def __init__(
        self, config, hidden_size, num_attention_heads, sequence_reduction_ratio
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(self, hidden_states, height, width, output_attentions=...):  # -> Any:
        ...

class GLPNDWConv(nn.Module):
    def __init__(self, dim=...) -> None: ...
    def forward(self, hidden_states, height, width):  # -> Any:
        ...

class GLPNMixFFN(nn.Module):
    def __init__(
        self, config, in_features, hidden_features=..., out_features=...
    ) -> None: ...
    def forward(self, hidden_states, height, width):  # -> Any:
        ...

class GLPNLayer(nn.Module):
    def __init__(
        self,
        config,
        hidden_size,
        num_attention_heads,
        drop_path,
        sequence_reduction_ratio,
        mlp_ratio,
    ) -> None: ...
    def forward(self, hidden_states, height, width, output_attentions=...):  # -> Any:
        ...

class GLPNEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        pixel_values,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...], ...] | BaseModelOutput:
        ...

@auto_docstring
class GLPNPreTrainedModel(PreTrainedModel):
    config: GLPNConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring
class GLPNModel(GLPNPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class GLPNSelectiveFeatureFusion(nn.Module):
    def __init__(self, in_channel=...) -> None: ...
    def forward(self, local_features, global_features): ...

class GLPNDecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels) -> None: ...
    def forward(self, hidden_state, residual=...):  # -> Any:
        ...

class GLPNDecoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: list[torch.Tensor]) -> list[torch.Tensor]: ...

class SiLogLoss(nn.Module):
    def __init__(self, lambd=...) -> None: ...
    def forward(self, pred, target):  # -> Tensor:
        ...

class GLPNDepthEstimationHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class GLPNForDepthEstimation(GLPNPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | DepthEstimatorOutput: ...

__all__ = ["GLPNForDepthEstimation", "GLPNLayer", "GLPNModel", "GLPNPreTrainedModel"]
