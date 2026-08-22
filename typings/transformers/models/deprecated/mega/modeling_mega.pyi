from typing import Any

from torch import nn

import torch

from .configuration_mega import MegaConfig
from ....cache_utils import Cache
from ....modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ....modeling_utils import PreTrainedModel
from ....utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ....utils.deprecation import deprecate_kwarg

"""PyTorch MEGA model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class MegaEmbeddings(nn.Module):
    def __init__(self, config: MegaConfig) -> None: ...
    def forward(self, input_ids=..., token_type_ids=..., inputs_embeds=...):  # -> None:
        ...

class MegaSimpleRelativePositionalBias(nn.Module):
    def __init__(self, config: MegaConfig) -> None: ...
    def forward(self, seq_len):  # -> Tensor:
        ...

class MegaRotaryRelativePositionalBias(nn.Module):
    def __init__(self, config: MegaConfig) -> None: ...
    @staticmethod
    def get_sinusoid_embeddings(
        max_positions: int, embedding_dim: int
    ):  # -> tuple[Tensor, Tensor]:
        ...
    def rotary(self, input):  # -> Tensor:
        ...
    def forward(self, seq_len):  # -> Tensor:
        ...

class MegaDropout(nn.Module):
    def __init__(self, dropout_probability, is_featurewise=...) -> None: ...
    def forward(self, input, batch_first: bool = ...):  # -> Tensor:
        ...

class MegaRMSNorm(nn.Module):
    def __init__(self, number_features, eps=..., affine=...) -> None: ...
    def forward(self, input): ...
    def extra_repr(self):  # -> str:
        ...

class MegaScaleNorm(nn.Module):
    def __init__(self, dim, eps=..., affine=...) -> None: ...
    def forward(self, input): ...

class MegaSequenceNorm(nn.Module):
    def __init__(
        self, norm_type, embedding_dim, eps=..., affine=..., export=...
    ) -> None: ...
    def forward(self, input):  # -> Any:
        ...

class MegaMultiDimensionDampedEma(nn.Module):
    def __init__(self, config: MegaConfig) -> None: ...
    def get_ema_coefficients(self):  # -> tuple[Tensor, Tensor]:
        ...
    def get_ema_kernel(self, length: int):  # -> Tensor:
        ...
    def fft_convolution(self, inputs, kernel, length): ...
    def ema_step(
        self, inputs, length, past_state=...
    ):  # -> tuple[Tensor, Any] | tuple[Any, Any | Tensor]:
        ...
    def one_ema_step(self, inputs, past_state=...):  # -> tuple[Tensor, Any]:
        ...
    def forward(
        self,
        inputs,
        attention_mask: torch.Tensor | None = ...,
        prev_state: torch.Tensor | None = ...,
        use_cache: bool = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MegaGatedCrossAttention(nn.Module):
    def __init__(self, config: MegaConfig) -> None: ...
    def element_attention(self, query, key, key_padding_mask, pidx): ...
    def softmax_attention(self, query, key, key_padding_mask, pidx):  # -> Any:
        ...
    def forward(
        self,
        query,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class MegaMovingAverageGatedAttention(nn.Module):
    def __init__(self, config: MegaConfig) -> None: ...
    def element_attention(self, query, key, padding_mask, causal_mask): ...
    def softmax_attention(self, query, key, padding_mask, causal_mask):  # -> Any:
        ...
    def forward(
        self,
        input,
        padding_mask: torch.Tensor | None = ...,
        causal_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions=...,
        use_cache=...,
    ):  # -> tuple[Any | Tensor, ...] | tuple[Any | Tensor, Any] | tuple[Any | Tensor]:
        ...

class MegaNormalizedFeedForwardNetwork(nn.Module):
    def __init__(self, config: MegaConfig) -> None: ...
    def forward(self, inputs):  # -> Any:
        ...

class MegaBlock(nn.Module):
    def __init__(self, config: MegaConfig) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = ...,
        causal_mask: torch.LongTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool = ...,
    ) -> tuple[torch.Tensor]: ...

class MegaPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MegaPreTrainedModel(PreTrainedModel):
    config: MegaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

MEGA_START_DOCSTRING = ...
MEGA_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    MEGA_START_DOCSTRING,
)
class MegaModel(MegaPreTrainedModel):
    def __init__(self, config: MegaConfig, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(
        MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...

@add_start_docstrings(
    ...,
    MEGA_START_DOCSTRING,
)
class MegaForCausalLM(MegaPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: MegaConfig) -> None: ...
    @add_start_docstrings_to_model_forward(
        MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithCrossAttentions: ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=..., attention_mask=..., **model_kwargs
    ):  # -> dict[str, Any | None]:
        ...

@add_start_docstrings(
    """MEGA Model with a `language modeling` head on top.""", MEGA_START_DOCSTRING
)
class MegaForMaskedLM(MegaPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: MegaConfig) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(
        MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | MaskedLMOutput: ...

@add_start_docstrings(
    ...,
    MEGA_START_DOCSTRING,
)
class MegaForSequenceClassification(MegaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

@add_start_docstrings(
    ...,
    MEGA_START_DOCSTRING,
)
class MegaForMultipleChoice(MegaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        MEGA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | MultipleChoiceModelOutput: ...

@add_start_docstrings(
    ...,
    MEGA_START_DOCSTRING,
)
class MegaForTokenClassification(MegaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput: ...

class MegaClassificationHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

@add_start_docstrings(
    ...,
    MEGA_START_DOCSTRING,
)
class MegaForQuestionAnswering(MegaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput: ...

__all__ = [
    "MegaForCausalLM",
    "MegaForMaskedLM",
    "MegaForMultipleChoice",
    "MegaForQuestionAnswering",
    "MegaForSequenceClassification",
    "MegaForTokenClassification",
    "MegaModel",
    "MegaPreTrainedModel",
]
