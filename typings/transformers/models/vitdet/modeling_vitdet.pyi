from torch import nn

import torch

from .configuration_vitdet import VitDetConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""PyTorch ViTDet backbone."""
logger = ...

class VitDetEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def get_absolute_positions(
        self, abs_pos_embeddings, has_cls_token, height, width
    ):  # -> Tensor:
        ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

@torch.jit.script_if_tracing
def get_rel_pos(q_size, k_size, rel_pos):  # -> Tensor:
    ...
def add_decomposed_relative_positions(
    attn, queries, rel_pos_h, rel_pos_w, q_size, k_size
): ...

class VitDetAttention(nn.Module):
    def __init__(self, config, input_size=...) -> None: ...
    def forward(
        self, hidden_state, output_attentions=...
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class VitDetDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class VitDetLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=...) -> None: ...
    def forward(self, x): ...

class VitDetResBottleneckBlock(nn.Module):
    def __init__(
        self, config, in_channels, out_channels, bottleneck_channels
    ) -> None: ...
    def forward(self, x): ...

class VitDetMlp(nn.Module):
    def __init__(self, config, in_features: int, hidden_features: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

def window_partition(hidden_state, window_size):  # -> tuple[Tensor, tuple[Any, Any]]:
    ...
def window_unpartition(windows, window_size, pad_height_width, height_width): ...

class VitDetLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: VitDetConfig,
        drop_path_rate: float = ...,
        window_size: int = ...,
        use_residual_block: bool = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class VitDetEncoder(nn.Module):
    def __init__(self, config: VitDetConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutput: ...

def caffe2_msra_fill(module: nn.Module) -> None: ...

@auto_docstring
class VitDetPreTrainedModel(PreTrainedModel):
    config: VitDetConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class VitDetModel(VitDetPreTrainedModel):
    def __init__(self, config: VitDetConfig) -> None: ...
    def get_input_embeddings(self) -> VitDetEmbeddings: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class VitDetBackbone(VitDetPreTrainedModel, BackboneMixin):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self) -> VitDetEmbeddings: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

__all__ = ["VitDetBackbone", "VitDetModel", "VitDetPreTrainedModel"]
