from torch import nn

import torch

from .configuration_segformer import SegformerConfig
from ...modeling_outputs import (
    BaseModelOutput,
    ImageClassifierOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch SegFormer model."""
logger = ...

class SegFormerImageClassifierOutput(ImageClassifierOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class SegformerDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class SegformerOverlapPatchEmbeddings(nn.Module):
    def __init__(self, patch_size, stride, num_channels, hidden_size) -> None: ...
    def forward(self, pixel_values):  # -> tuple[Any, Any, Any]:
        ...

class SegformerEfficientSelfAttention(nn.Module):
    def __init__(
        self, config, hidden_size, num_attention_heads, sequence_reduction_ratio
    ) -> None: ...
    def forward(
        self, hidden_states, height, width, output_attentions=...
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class SegformerSelfOutput(nn.Module):
    def __init__(self, config, hidden_size) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

class SegformerAttention(nn.Module):
    def __init__(
        self, config, hidden_size, num_attention_heads, sequence_reduction_ratio
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(self, hidden_states, height, width, output_attentions=...):  # -> Any:
        ...

class SegformerDWConv(nn.Module):
    def __init__(self, dim=...) -> None: ...
    def forward(self, hidden_states, height, width):  # -> Any:
        ...

class SegformerMixFFN(nn.Module):
    def __init__(
        self, config, in_features, hidden_features=..., out_features=...
    ) -> None: ...
    def forward(self, hidden_states, height, width):  # -> Any:
        ...

class SegformerLayer(nn.Module):
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

class SegformerEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring
class SegformerPreTrainedModel(PreTrainedModel):
    config: SegformerConfig
    base_model_prefix = ...
    main_input_name = ...

@auto_docstring
class SegformerModel(SegformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class SegformerForImageClassification(SegformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SegFormerImageClassifierOutput: ...

class SegformerMLP(nn.Module):
    def __init__(self, config: SegformerConfig, input_dim) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class SegformerDecodeHead(SegformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class SegformerForSemanticSegmentation(SegformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SemanticSegmenterOutput: ...

__all__ = [
    "SegformerDecodeHead",
    "SegformerForImageClassification",
    "SegformerForSemanticSegmentation",
    "SegformerLayer",
    "SegformerModel",
    "SegformerPreTrainedModel",
]
