from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_funnel import FunnelConfig
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch Funnel Transformer model."""
logger = ...
INF = ...

def load_tf_weights_in_funnel(model, config, tf_checkpoint_path): ...

class FunnelEmbeddings(nn.Module):
    def __init__(self, config: FunnelConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class FunnelAttentionStructure(nn.Module):
    cls_token_type_id: int = ...
    def __init__(self, config: FunnelConfig) -> None: ...
    def init_attention_inputs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def token_type_ids_to_mat(self, token_type_ids: torch.Tensor) -> torch.Tensor: ...
    def get_position_embeds(
        self, seq_len: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor] | list[list[torch.Tensor]]: ...
    def stride_pool_pos(self, pos_id: torch.Tensor, block_index: int):  # -> Tensor:
        ...
    def relative_pos(
        self, pos: torch.Tensor, stride: int, pooled_pos=..., shift: int = ...
    ) -> torch.Tensor: ...
    def stride_pool(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor],
        axis: int | tuple[int] | list[int],
    ) -> torch.Tensor: ...
    def pool_tensor(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor],
        mode: str = ...,
        stride: int = ...,
    ) -> torch.Tensor: ...
    def pre_attention_pooling(
        self, output, attention_inputs: tuple[torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]: ...
    def post_attention_pooling(
        self, attention_inputs: tuple[torch.Tensor]
    ) -> tuple[torch.Tensor]: ...

class FunnelRelMultiheadAttention(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int) -> None: ...
    def relative_positional_attention(
        self, position_embeds, q_head, context_len, cls_mask=...
    ):  # -> Tensor:
        ...
    def relative_token_type_attention(
        self, token_type_mat, q_head, cls_mask=...
    ):  # -> Tensor | Literal[0]:
        ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_inputs: tuple[torch.Tensor],
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, ...]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]: ...

class FunnelPositionwiseFFN(nn.Module):
    def __init__(self, config: FunnelConfig) -> None: ...
    def forward(self, hidden: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class FunnelLayer(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int) -> None: ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_inputs,
        output_attentions: bool = ...,
    ) -> tuple: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple: ...

class FunnelEncoder(nn.Module):
    def __init__(self, config: FunnelConfig) -> None: ...
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

def upsample(
    x: torch.Tensor,
    stride: int,
    target_len: int,
    separate_cls: bool = ...,
    truncate_seq: bool = ...,
) -> torch.Tensor: ...

class FunnelDecoder(nn.Module):
    def __init__(self, config: FunnelConfig) -> None: ...
    def forward(
        self,
        final_hidden: torch.Tensor,
        first_block_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

class FunnelDiscriminatorPredictions(nn.Module):
    def __init__(self, config: FunnelConfig) -> None: ...
    def forward(self, discriminator_hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class FunnelPreTrainedModel(PreTrainedModel):
    config: FunnelConfig
    load_tf_weights = ...
    base_model_prefix = ...

class FunnelClassificationHead(nn.Module):
    def __init__(self, config: FunnelConfig, n_labels: int) -> None: ...
    def forward(self, hidden: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`FunnelForPreTraining`].
    """
)
class FunnelForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@auto_docstring(custom_intro=...)
class FunnelBaseModel(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Embedding: ...
    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring
class FunnelModel(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Embedding: ...
    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class FunnelForPreTraining(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | FunnelForPreTrainingOutput: ...

@auto_docstring
class FunnelForMaskedLM(FunnelPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: FunnelConfig) -> None: ...
    def get_output_embeddings(self) -> nn.Linear: ...
    def set_output_embeddings(self, new_embeddings: nn.Embedding) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class FunnelForSequenceClassification(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

@auto_docstring
class FunnelForMultipleChoice(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MultipleChoiceModelOutput: ...

@auto_docstring
class FunnelForTokenClassification(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class FunnelForQuestionAnswering(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "FunnelBaseModel",
    "FunnelForMaskedLM",
    "FunnelForMultipleChoice",
    "FunnelForPreTraining",
    "FunnelForQuestionAnswering",
    "FunnelForSequenceClassification",
    "FunnelForTokenClassification",
    "FunnelModel",
    "FunnelPreTrainedModel",
    "load_tf_weights_in_funnel",
]
