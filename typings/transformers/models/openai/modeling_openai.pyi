from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_openai import OpenAIGPTConfig
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch OpenAI GPT model."""
logger = ...

def load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path): ...

ACT_FNS = ...

class Attention(nn.Module):
    def __init__(self, nx, n_positions, config, scale=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def merge_heads(self, x): ...
    def split_heads(self, x, k=...): ...
    def forward(
        self, x, attention_mask=..., head_mask=..., output_attentions=...
    ):  # -> list[Any]:
        ...

class MLP(nn.Module):
    def __init__(self, n_state, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Block(nn.Module):
    def __init__(self, n_positions, config, scale=...) -> None: ...
    def forward(
        self, x, attention_mask=..., head_mask=..., output_attentions=...
    ):  # -> Any:
        ...

class OpenAIGPTSequenceSummary(nn.Module):
    def __init__(self, config: OpenAIGPTConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        cls_index: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

@auto_docstring
class OpenAIGPTPreTrainedModel(PreTrainedModel):
    config: OpenAIGPTConfig
    load_tf_weights = ...
    base_model_prefix = ...

@dataclass
@auto_docstring(custom_intro=...)
class OpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    mc_loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    mc_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | CausalLMOutput: ...
    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, **kwargs
    ) -> dict[str, Any]: ...

@auto_docstring(custom_intro=...)
class OpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        mc_token_ids: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        mc_labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | OpenAIGPTDoubleHeadsModelOutput: ...

@auto_docstring(custom_intro=...)
class OpenAIGPTForSequenceClassification(OpenAIGPTPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

__all__ = [
    "OpenAIGPTDoubleHeadsModel",
    "OpenAIGPTForSequenceClassification",
    "OpenAIGPTLMHeadModel",
    "OpenAIGPTModel",
    "OpenAIGPTPreTrainedModel",
    "load_tf_weights_in_openai_gpt",
]
