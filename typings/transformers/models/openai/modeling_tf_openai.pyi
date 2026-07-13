from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_openai import OpenAIGPTConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFCausalLMOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 OpenAI GPT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class TFAttention(keras.layers.Layer):
    def __init__(self, nx, config, scale=..., **kwargs) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    @staticmethod
    def causal_attention_mask(nd, ns): ...
    def merge_heads(self, x): ...
    def split_heads(self, x): ...
    def call(
        self, x, attention_mask, head_mask, output_attentions, training=...
    ):  # -> list[Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMLP(keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs) -> None: ...
    def call(self, x, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlock(keras.layers.Layer):
    def __init__(self, config, scale=..., **kwargs) -> None: ...
    def call(self, x, attention_mask, head_mask, output_attentions, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFOpenAIGPTMainLayer(keras.layers.Layer):
    config_class = OpenAIGPTConfig
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def get_input_embeddings(self):  # -> TFSharedEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutput: ...

class TFOpenAIGPTPreTrainedModel(TFPreTrainedModel):
    config_class = OpenAIGPTConfig
    base_model_prefix = ...

@dataclass
class TFOpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    logits: tf.Tensor | None = ...
    mc_logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

OPENAI_GPT_START_DOCSTRING = ...
OPENAI_GPT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTLMHeadModel(TFOpenAIGPTPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_output_embeddings(self): ...
    def set_output_embeddings(self, value):  # -> None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFCausalLMOutput: ...
    def prepare_inputs_for_generation(self, inputs, **kwargs):  # -> dict[str, Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTDoubleHeadsModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFOpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        mc_token_ids: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFOpenAIGPTDoubleHeadsModelOutput: ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTForSequenceClassification(
    TFOpenAIGPTPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFSequenceClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFOpenAIGPTDoubleHeadsModel",
    "TFOpenAIGPTForSequenceClassification",
    "TFOpenAIGPTLMHeadModel",
    "TFOpenAIGPTMainLayer",
    "TFOpenAIGPTModel",
    "TFOpenAIGPTPreTrainedModel",
]
