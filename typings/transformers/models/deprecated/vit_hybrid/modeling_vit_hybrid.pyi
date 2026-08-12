from typing import Any
from torch import nn

import torch

from .configuration_vit_hybrid import ViTHybridConfig
from ....modeling_layers import GradientCheckpointingLayer
from ....modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from ....modeling_utils import PreTrainedModel
from ....utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""PyTorch ViT Hybrid model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

class ViTHybridEmbeddings(nn.Module):
    def __init__(self, config: ViTHybridConfig, use_mask_token: bool = ...) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ViTHybridPatchEmbeddings(nn.Module):
    def __init__(self, config, feature_size=...) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ViTHybridSelfAttention(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        hidden_states,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class ViTHybridSdpaSelfAttention(ViTHybridSelfAttention):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    def forward(
        self,
        hidden_states,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class ViTHybridSelfOutput(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ViTHybridAttention(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    def prune_heads(self, heads: set[int]) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class ViTHybridSdpaAttention(ViTHybridAttention):
    def __init__(self, config: ViTHybridConfig) -> None: ...

class ViTHybridIntermediate(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ViTHybridOutput(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

VIT_HYBRID_ATTENTION_CLASSES = ...

class ViTHybridLayer(GradientCheckpointingLayer):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class ViTHybridEncoder(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

class ViTHybridPreTrainedModel(PreTrainedModel):
    config: ViTHybridConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...

VIT_START_DOCSTRING = ...
VIT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    VIT_START_DOCSTRING,
)
class ViTHybridModel(ViTHybridPreTrainedModel):
    def __init__(
        self,
        config: ViTHybridConfig,
        add_pooling_layer: bool = ...,
        use_mask_token: bool = ...,
    ) -> None: ...
    def get_input_embeddings(self) -> ViTHybridPatchEmbeddings: ...
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class ViTHybridPooler(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

@add_start_docstrings(
    ...,
    VIT_START_DOCSTRING,
)
class ViTHybridForImageClassification(ViTHybridPreTrainedModel):
    def __init__(self, config: ViTHybridConfig) -> None: ...
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutput: ...

__all__ = [
    "ViTHybridForImageClassification",
    "ViTHybridModel",
    "ViTHybridPreTrainedModel",
]
