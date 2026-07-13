from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_funnel import FunnelConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
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

"""TF 2.0 Funnel model."""
logger = ...
_CONFIG_FOR_DOC = ...
INF = ...

class TFFunnelEmbeddings(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(self, input_ids=..., inputs_embeds=..., training=...): ...

class TFFunnelAttentionStructure:
    cls_token_type_id: int = ...
    def __init__(self, config) -> None: ...
    def init_attention_inputs(
        self, inputs_embeds, attention_mask=..., token_type_ids=..., training=...
    ):  # -> tuple[tuple[Any, Any, Any, Any] | list[Any], Any | None, Any | None, Any | None]:
        ...
    def token_type_ids_to_mat(self, token_type_ids): ...
    def get_position_embeds(
        self, seq_len, training=...
    ):  # -> tuple[Any, Any, Any, Any] | list[Any]:
        ...
    def stride_pool_pos(self, pos_id, block_index): ...
    def relative_pos(self, pos, stride, pooled_pos=..., shift=...): ...
    def stride_pool(self, tensor, axis):  # -> list[Any] | tuple[Any, ...] | None:
        ...
    def pool_tensor(
        self, tensor, mode=..., stride=...
    ):  # -> list[Any] | tuple[Any, ...] | None:
        ...
    def pre_attention_pooling(
        self, output, attention_inputs
    ):  # -> tuple[list[Any] | tuple[Any, ...] | Any | None, tuple[Any | list[Any] | tuple[Any, ...] | None, Any | list[Any] | tuple[Any, ...] | None, Any | list[Any] | tuple[Any, ...] | None, Any | list[Any] | tuple[Any, ...] | None]]:
        ...
    def post_attention_pooling(
        self, attention_inputs
    ):  # -> tuple[Any, Any | list[Any] | tuple[Any, ...] | None, list[Any] | tuple[Any, ...] | Any | None, Any | list[Any] | tuple[Any, ...] | None]:
        ...

class TFFunnelRelMultiheadAttention(keras.layers.Layer):
    def __init__(self, config, block_index, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def relative_positional_attention(
        self, position_embeds, q_head, context_len, cls_mask=...
    ): ...
    def relative_token_type_attention(
        self, token_type_mat, q_head, cls_mask=...
    ):  # -> Literal[0]:
        ...
    def call(
        self, query, key, value, attention_inputs, output_attentions=..., training=...
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class TFFunnelPositionwiseFFN(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFFunnelLayer(keras.layers.Layer):
    def __init__(self, config, block_index, **kwargs) -> None: ...
    def call(
        self, query, key, value, attention_inputs, output_attentions=..., training=...
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFFunnelEncoder(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self,
        inputs_embeds,
        attention_mask=...,
        token_type_ids=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> tuple[Any | tuple[Any] | tuple[Any, ...] | tuple[()], ...] | TFBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

def upsample(x, stride, target_len, separate_cls=..., truncate_seq=...): ...

class TFFunnelDecoder(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self,
        final_hidden,
        first_block_hidden,
        attention_mask=...,
        token_type_ids=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> tuple[Any | tuple[Any] | tuple[Any, ...] | tuple[()], ...] | TFBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFFunnelBaseLayer(keras.layers.Layer):
    config_class = FunnelConfig
    def __init__(self, config, **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFFunnelEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        attention_mask=...,
        token_type_ids=...,
        inputs_embeds=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFFunnelMainLayer(keras.layers.Layer):
    config_class = FunnelConfig
    def __init__(self, config, **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFFunnelEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        attention_mask=...,
        token_type_ids=...,
        inputs_embeds=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> tuple[Any, ...] | tuple[Any, Any] | tuple[Any] | TFBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFFunnelDiscriminatorPredictions(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, discriminator_hidden_states): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFFunnelMaskedLMHead(keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Any:
        ...
    def set_output_embeddings(self, value):  # -> None:
        ...
    def get_bias(self):  # -> dict[str, Any]:
        ...
    def set_bias(self, value):  # -> None:
        ...
    def call(self, hidden_states, training=...): ...

class TFFunnelClassificationHead(keras.layers.Layer):
    def __init__(self, config, n_labels, **kwargs) -> None: ...
    def call(self, hidden, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFFunnelPreTrainedModel(TFPreTrainedModel):
    config_class = FunnelConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any]:
        ...

@dataclass
class TFFunnelForPreTrainingOutput(ModelOutput):
    logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

FUNNEL_START_DOCSTRING = ...
FUNNEL_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelBaseModel(TFFunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None: ...
    @add_start_docstrings_to_model_forward(
        FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFBaseModelOutput: ...
    def serving_output(self, output):  # -> TFBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelModel(TFFunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFBaseModelOutput: ...
    def serving_output(self, output):  # -> TFBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForPreTraining(TFFunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFFunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs,
    ) -> tuple[tf.Tensor] | TFFunnelForPreTrainingOutput: ...
    def serving_output(self, output):  # -> TFFunnelForPreTrainingOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(..., FUNNEL_START_DOCSTRING)
class TFFunnelForMaskedLM(TFFunnelPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> TFFunnelMaskedLMHead: ...
    def get_prefix_bias_name(self) -> str: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFMaskedLMOutput: ...
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForSequenceClassification(
    TFFunnelPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFSequenceClassifierOutput: ...
    def serving_output(
        self, output: TFSequenceClassifierOutput
    ) -> TFSequenceClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForMultipleChoice(TFFunnelPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None: ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any]:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FUNNEL_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFMultipleChoiceModelOutput: ...
    def serving_output(
        self, output: TFMultipleChoiceModelOutput
    ) -> TFMultipleChoiceModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForTokenClassification(
    TFFunnelPreTrainedModel, TFTokenClassificationLoss
):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFTokenClassifierOutput: ...
    def serving_output(
        self, output: TFTokenClassifierOutput
    ) -> TFTokenClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForQuestionAnswering(TFFunnelPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        start_positions: np.ndarray | tf.Tensor | None = ...,
        end_positions: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFQuestionAnsweringModelOutput: ...
    def serving_output(
        self, output: TFQuestionAnsweringModelOutput
    ) -> TFQuestionAnsweringModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFFunnelBaseModel",
    "TFFunnelForMaskedLM",
    "TFFunnelForMultipleChoice",
    "TFFunnelForPreTraining",
    "TFFunnelForQuestionAnswering",
    "TFFunnelForSequenceClassification",
    "TFFunnelForTokenClassification",
    "TFFunnelModel",
    "TFFunnelPreTrainedModel",
]
