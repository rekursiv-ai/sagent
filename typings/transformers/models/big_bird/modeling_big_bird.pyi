from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_big_bird import BigBirdConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch BigBird model."""
logger = ...
_TRIVIA_QA_MAPPING = ...

def load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=...): ...

class BigBirdEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids=...,
        token_type_ids=...,
        position_ids=...,
        inputs_embeds=...,
        past_key_values_length=...,
    ):  # -> Any:
        ...

class BigBirdSelfAttention(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_values=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Tensor, Any]:
        ...

class BigBirdBlockSparseAttention(nn.Module):
    def __init__(self, config, seed=...) -> None: ...
    def forward(
        self,
        hidden_states,
        band_mask=...,
        from_mask=...,
        to_mask=...,
        from_blocked_mask=...,
        to_blocked_mask=...,
        output_attentions=...,
    ):  # -> tuple[Tensor, Tensor | None]:
        ...
    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=...):  # -> Tensor:
        ...
    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=...):  # -> Tensor:
        ...
    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        n_rand_blocks,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_len,
        to_seq_len,
        seed,
        plan_from_length,
        plan_num_rand_blocks,
        output_attentions,
    ):  # -> tuple[Tensor, Tensor | None]:
        ...
    @staticmethod
    def torch_gather_b2(params, indices): ...

class BigBirdSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BigBirdAttention(nn.Module):
    def __init__(self, config, seed=...) -> None: ...
    def set_attention_type(self, value: str, layer_idx=...):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_values=...,
        output_attentions=...,
        band_mask=...,
        from_mask=...,
        to_mask=...,
        from_blocked_mask=...,
        to_blocked_mask=...,
        cache_position=...,
    ):  # -> Any:
        ...

class BigBirdIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BigBirdOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BigBirdLayer(GradientCheckpointingLayer):
    def __init__(self, config, seed=...) -> None: ...
    def set_attention_type(self, value: str, layer_idx=...):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        band_mask=...,
        from_mask=...,
        to_mask=...,
        blocked_encoder_mask=...,
        past_key_values=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class BigBirdEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def set_attention_type(self, value: str):  # -> None:
        ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        band_mask=...,
        from_mask=...,
        to_mask=...,
        blocked_encoder_mask=...,
        return_dict=...,
        cache_position=...,
    ) -> BaseModelOutputWithPastAndCrossAttentions | tuple: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> BaseModelOutputWithPastAndCrossAttentions | tuple: ...

class BigBirdPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BigBirdLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class BigBirdOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BigBirdOnlyNSPHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pooled_output):  # -> Any:
        ...

class BigBirdPreTrainingHeads(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output, pooled_output):  # -> tuple[Any, Any]:
        ...

@auto_docstring
class BigBirdPreTrainedModel(PreTrainedModel):
    config: BigBirdConfig
    load_tf_weights = ...
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

@dataclass
@auto_docstring(custom_intro=...)
class BigBirdForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_logits: torch.FloatTensor | None = ...
    seq_relationship_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class BigBirdForQuestionAnsweringModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_logits: torch.FloatTensor | None = ...
    end_logits: torch.FloatTensor | None = ...
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class BigBirdModel(BigBirdPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_attention_type(self, value: str):  # -> None:
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
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions | tuple[torch.FloatTensor]: ...
    @staticmethod
    def create_masks_for_block_sparse_attn(
        attention_mask: torch.Tensor, block_size: int
    ):  # -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ...

class BigBirdForPreTraining(BigBirdPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
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
        labels: torch.FloatTensor | None = ...,
        next_sentence_label: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BigBirdForPreTrainingOutput | tuple[torch.FloatTensor]: ...

@auto_docstring
class BigBirdForMaskedLM(BigBirdPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
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
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> MaskedLMOutput | tuple[torch.FloatTensor]: ...
    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=..., **model_kwargs
    ):  # -> dict[str, Tensor | Any]:
        ...

@auto_docstring(custom_intro=...)
class BigBirdForCausalLM(BigBirdPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
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
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions | tuple[torch.FloatTensor]: ...

class BigBirdClassificationHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class BigBirdForSequenceClassification(BigBirdPreTrainedModel):
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
    ) -> SequenceClassifierOutput | tuple[torch.FloatTensor]: ...

@auto_docstring
class BigBirdForMultipleChoice(BigBirdPreTrainedModel):
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
    ) -> MultipleChoiceModelOutput | tuple[torch.FloatTensor]: ...

@auto_docstring
class BigBirdForTokenClassification(BigBirdPreTrainedModel):
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
    ) -> TokenClassifierOutput | tuple[torch.FloatTensor]: ...

class BigBirdForQuestionAnsweringHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, encoder_output):  # -> Any:
        ...

@auto_docstring
class BigBirdForQuestionAnswering(BigBirdPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        question_lengths: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BigBirdForQuestionAnsweringModelOutput | tuple[torch.FloatTensor]: ...
    @staticmethod
    def prepare_question_mask(q_lengths: torch.Tensor, maxlen: int):  # -> Tensor:
        ...

__all__ = [
    "BigBirdForCausalLM",
    "BigBirdForMaskedLM",
    "BigBirdForMultipleChoice",
    "BigBirdForPreTraining",
    "BigBirdForQuestionAnswering",
    "BigBirdForSequenceClassification",
    "BigBirdForTokenClassification",
    "BigBirdLayer",
    "BigBirdModel",
    "BigBirdPreTrainedModel",
    "load_tf_weights_in_big_bird",
]
