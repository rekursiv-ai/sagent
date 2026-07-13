from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_transfo_xl import TransfoXLConfig
from ....modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ....utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""
TF 2.0 Transformer XL model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class TFPositionalEmbedding(keras.layers.Layer):
    def __init__(self, demb, **kwargs) -> None: ...
    def call(self, pos_seq, bsz=...): ...

class TFPositionwiseFF(keras.layers.Layer):
    def __init__(
        self,
        d_model,
        d_inner,
        dropout,
        pre_lnorm=...,
        layer_norm_epsilon=...,
        init_std=...,
        **kwargs,
    ) -> None: ...
    def call(self, inp, training=...): ...

class TFRelPartialLearnableMultiHeadAttn(keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=...,
        pre_lnorm=...,
        r_r_bias=...,
        r_w_bias=...,
        layer_norm_epsilon=...,
        init_std=...,
        output_attentions=...,
        **kwargs,
    ) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(
        self, w, r, attn_mask, mems, head_mask, output_attentions, training=...
    ):  # -> list[Any]:
        ...

class TFRelPartialLearnableDecoderLayer(keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt=...,
        pre_lnorm=...,
        r_w_bias=...,
        r_r_bias=...,
        layer_norm_epsilon=...,
        init_std=...,
        output_attentions=...,
        **kwargs,
    ) -> None: ...
    def call(
        self,
        dec_inp,
        r,
        dec_attn_mask,
        mems,
        head_mask,
        output_attentions,
        training=...,
    ): ...

class TFTransfoEmbeddings(keras.layers.Layer):
    def __init__(self, vocab_size, emb_size, init_std, **kwargs) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, inputs): ...

class TFAdaptiveEmbedding(keras.layers.Layer):
    def __init__(
        self,
        n_token,
        d_embed,
        d_proj,
        cutoffs,
        div_val=...,
        init_std=...,
        sample_softmax=...,
        **kwargs,
    ) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, inp): ...

@keras_serializable
class TFTransfoXLMainLayer(keras.layers.Layer):
    config_class = TransfoXLConfig
    def __init__(self, config, **kwargs) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def get_input_embeddings(self):  # -> TFAdaptiveEmbedding:
        ...
    def set_input_embeddings(self, value): ...
    def backward_compatible(self):  # -> None:
        ...
    def reset_memory_length(self, mem_len):  # -> None:
        ...
    def init_mems(self, bsz):  # -> list[Any] | None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        mems: list[tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ):  # -> tuple[Any | list[Any] | tuple[*tuple[Any, ...], Any] | tuple[Any, ...], ...] | TFTransfoXLModelOutput:
        ...

class TFTransfoXLPreTrainedModel(TFPreTrainedModel):
    config_class = TransfoXLConfig
    base_model_prefix = ...

@dataclass
class TFTransfoXLModelOutput(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    mems: list[tf.Tensor] = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

@dataclass
class TFTransfoXLLMHeadModelOutput(ModelOutput):
    prediction_scores: tf.Tensor | None = ...
    mems: list[tf.Tensor] = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

@dataclass
class TFTransfoXLSequenceClassifierOutputWithPast(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    mems: list[tf.Tensor] = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

TRANSFO_XL_START_DOCSTRING = ...
TRANSFO_XL_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLModel(TFTransfoXLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        mems: list[tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFTransfoXLModelOutput | tuple[tf.Tensor]: ...

@add_start_docstrings(
    ...,
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLLMHeadModel(TFTransfoXLPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> None:
        ...
    def reset_memory_length(self, mem_len):  # -> None:
        ...
    def init_mems(self, bsz):  # -> list[Any] | None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        mems: list[tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFTransfoXLLMHeadModelOutput | tuple[tf.Tensor]: ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=..., **model_kwargs
    ):  # -> dict[Any, Any]:
        ...
    def tf_to_pt_weight_rename(
        self, tf_weight
    ):  # -> tuple[Any, Any] | tuple[Any] | None:
        ...

@add_start_docstrings(
    ...,
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLForSequenceClassification(
    TFTransfoXLPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_output_embeddings(self):  # -> TFAdaptiveEmbedding:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLSequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        mems: list[tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFTransfoXLSequenceClassifierOutputWithPast: ...

__all__ = [
    "TFAdaptiveEmbedding",
    "TFTransfoXLForSequenceClassification",
    "TFTransfoXLLMHeadModel",
    "TFTransfoXLMainLayer",
    "TFTransfoXLModel",
    "TFTransfoXLPreTrainedModel",
]
