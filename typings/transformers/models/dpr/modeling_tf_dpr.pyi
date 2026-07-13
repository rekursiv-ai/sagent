from dataclasses import dataclass

import tensorflow as tf

from .configuration_dpr import DPRConfig
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TensorFlow DPR model for Open Domain Question Answering."""
logger = ...
_CONFIG_FOR_DOC = ...

@dataclass
class TFDPRContextEncoderOutput(ModelOutput):
    pooler_output: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFDPRQuestionEncoderOutput(ModelOutput):
    pooler_output: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFDPRReaderOutput(ModelOutput):
    start_logits: tf.Tensor | None = ...
    end_logits: tf.Tensor | None = ...
    relevance_logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

class TFDPREncoderLayer(keras.layers.Layer):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor, ...]: ...
    @property
    def embeddings_size(self) -> int: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDPRSpanPredictorLayer(keras.layers.Layer):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        training: bool = ...,
    ) -> TFDPRReaderOutput | tuple[tf.Tensor, ...]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDPRSpanPredictor(TFPreTrainedModel):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        training: bool = ...,
    ) -> TFDPRReaderOutput | tuple[tf.Tensor, ...]: ...

class TFDPREncoder(TFPreTrainedModel):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        training: bool = ...,
    ) -> TFDPRReaderOutput | tuple[tf.Tensor, ...]: ...

class TFDPRPretrainedContextEncoder(TFPreTrainedModel):
    config_class = DPRConfig
    base_model_prefix = ...

class TFDPRPretrainedQuestionEncoder(TFPreTrainedModel):
    config_class = DPRConfig
    base_model_prefix = ...

class TFDPRPretrainedReader(TFPreTrainedModel):
    config_class = DPRConfig
    base_model_prefix = ...

TF_DPR_START_DOCSTRING = ...
TF_DPR_ENCODERS_INPUTS_DOCSTRING = ...
TF_DPR_READER_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    TF_DPR_START_DOCSTRING,
)
class TFDPRContextEncoder(TFDPRPretrainedContextEncoder):
    def __init__(self, config: DPRConfig, *args, **kwargs) -> None: ...
    def get_input_embeddings(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFDPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFDPRContextEncoderOutput | tuple[tf.Tensor, ...]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    TF_DPR_START_DOCSTRING,
)
class TFDPRQuestionEncoder(TFDPRPretrainedQuestionEncoder):
    def __init__(self, config: DPRConfig, *args, **kwargs) -> None: ...
    def get_input_embeddings(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFDPRQuestionEncoderOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFDPRQuestionEncoderOutput | tuple[tf.Tensor, ...]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    TF_DPR_START_DOCSTRING,
)
class TFDPRReader(TFDPRPretrainedReader):
    def __init__(self, config: DPRConfig, *args, **kwargs) -> None: ...
    def get_input_embeddings(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_READER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFDPRReaderOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFDPRReaderOutput | tuple[tf.Tensor, ...]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFDPRContextEncoder",
    "TFDPRPretrainedContextEncoder",
    "TFDPRPretrainedQuestionEncoder",
    "TFDPRPretrainedReader",
    "TFDPRQuestionEncoder",
    "TFDPRReader",
]
