from torch import nn

import torch

from .configuration_van import VanConfig
from ....modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ....modeling_utils import PreTrainedModel
from ....utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""PyTorch Visual Attention Network (VAN) model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class VanDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class VanOverlappingPatchEmbedder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int = ...,
        stride: int = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

class VanMlpLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        hidden_act: str = ...,
        dropout_rate: float = ...,
    ) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...

class VanLargeKernelAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...

class VanLargeKernelAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...

class VanSpatialAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, hidden_act: str = ...) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...

class VanLayerScaling(nn.Module):
    def __init__(self, hidden_size: int, initial_value: float = ...) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...

class VanLayer(nn.Module):
    def __init__(
        self,
        config: VanConfig,
        hidden_size: int,
        mlp_ratio: int = ...,
        drop_path_rate: float = ...,
    ) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...

class VanStage(nn.Module):
    def __init__(
        self,
        config: VanConfig,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        stride: int,
        depth: int,
        mlp_ratio: int = ...,
        drop_path_rate: float = ...,
    ) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...

class VanEncoder(nn.Module):
    def __init__(self, config: VanConfig) -> None: ...
    def forward(
        self,
        hidden_state: torch.Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithNoAttention: ...

class VanPreTrainedModel(PreTrainedModel):
    config: VanConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

VAN_START_DOCSTRING = ...
VAN_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    VAN_START_DOCSTRING,
)
class VanModel(VanPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor | None,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPoolingAndNoAttention: ...

@add_start_docstrings(
    ...,
    VAN_START_DOCSTRING,
)
class VanForImageClassification(VanPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutputWithNoAttention: ...

__all__ = ["VanForImageClassification", "VanModel", "VanPreTrainedModel"]
