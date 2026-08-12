from typing import Any
from collections.abc import Iterable, Iterator

from torch import nn

import torch

from .configuration_graphormer import GraphormerConfig
from ....modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)
from ....modeling_utils import PreTrainedModel

"""PyTorch Graphormer model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

def quant_noise(
    module: nn.Module, p: float, block_size: int
):  # -> Module | Linear | Embedding | Conv2d:
    ...

class LayerDropModuleList(nn.ModuleList):
    def __init__(self, p: float, modules: Iterable[nn.Module] | None = ...) -> None: ...
    def __iter__(self) -> Iterator[nn.Module]: ...

class GraphormerGraphNodeFeature(nn.Module):
    def __init__(self, config: GraphormerConfig) -> None: ...
    def forward(
        self,
        input_nodes: torch.LongTensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class GraphormerGraphAttnBias(nn.Module):
    def __init__(self, config: GraphormerConfig) -> None: ...
    def forward(
        self,
        input_nodes: torch.LongTensor,
        attn_bias: torch.Tensor,
        spatial_pos: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class GraphormerMultiheadAttention(nn.Module):
    def __init__(self, config: GraphormerConfig) -> None: ...
    def reset_parameters(self):  # -> None:
        ...
    def forward(
        self,
        query: torch.LongTensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        attn_bias: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None = ...,
        need_weights: bool = ...,
        attn_mask: torch.Tensor | None = ...,
        before_softmax: bool = ...,
        need_head_weights: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def apply_sparse_mask(
        self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int
    ) -> torch.Tensor: ...

class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(self, config: GraphormerConfig) -> None: ...
    def build_fc(
        self, input_dim: int, output_dim: int, q_noise: float, qn_block_size: int
    ) -> nn.Module | nn.Linear | nn.Embedding | nn.Conv2d: ...
    def forward(
        self,
        input_nodes: torch.Tensor,
        self_attn_bias: torch.Tensor | None = ...,
        self_attn_mask: torch.Tensor | None = ...,
        self_attn_padding_mask: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class GraphormerGraphEncoder(nn.Module):
    def __init__(self, config: GraphormerConfig) -> None: ...
    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        perturb=...,
        last_state_only: bool = ...,
        token_embeddings: torch.Tensor | None = ...,
        attn_mask: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor | list[torch.LongTensor], torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor | list[torch.LongTensor], torch.Tensor]: ...

class GraphormerDecoderHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None: ...
    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class GraphormerPreTrainedModel(PreTrainedModel):
    config: GraphormerConfig
    base_model_prefix = ...
    main_input_name_nodes = ...
    main_input_name_edges = ...
    def normal_(self, data: torch.Tensor):  # -> None:
        ...
    def init_graphormer_params(
        self, module: nn.Linear | nn.Embedding | GraphormerMultiheadAttention
    ):  # -> None:
        ...

class GraphormerModel(GraphormerPreTrainedModel):
    def __init__(self, config: GraphormerConfig) -> None: ...
    def reset_output_layer_parameters(self):  # -> None:
        ...
    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        perturb: torch.FloatTensor | None = ...,
        masked_tokens: None = ...,
        return_dict: bool | None = ...,
        **unused,
    ) -> tuple[torch.LongTensor] | BaseModelOutputWithNoAttention: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.LongTensor] | BaseModelOutputWithNoAttention: ...
    def max_nodes(self):  # -> Callable[[], ...]:
        ...

class GraphormerForGraphClassification(GraphormerPreTrainedModel):
    def __init__(self, config: GraphormerConfig) -> None: ...
    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        labels: torch.LongTensor | None = ...,
        return_dict: bool | None = ...,
        **unused,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

__all__ = [
    "GraphormerForGraphClassification",
    "GraphormerModel",
    "GraphormerPreTrainedModel",
]
