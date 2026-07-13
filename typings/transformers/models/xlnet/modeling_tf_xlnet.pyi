from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_xlnet import XLNetConfig
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
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

"""
TF 2.0 XLNet model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class TFXLNetRelativeAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def prune_heads(self, heads): ...
    def rel_shift(self, x, klen=...): ...
    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        seg_mat,
        attn_mask,
        head_mask,
        output_attentions,
        training=...,
    ):  # -> tuple[Any, Any]:
        ...
    def post_attention(self, h, attn_vec, residual=..., training=...): ...
    def call(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems: np.ndarray | tf.Tensor | None = ...,
        target_mapping: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        training: bool = ...,
    ):  # -> tuple[Any, Any | None, tuple[Any, Any] | Any] | tuple[Any, Any | None]:
        ...

class TFXLNetFeedForward(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, inp, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFXLNetLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self,
        output_h,
        output_g,
        non_tgt_mask,
        attn_mask,
        pos_emb,
        seg_mat,
        mems: np.ndarray | tf.Tensor | None = ...,
        target_mapping: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        training: bool = ...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFXLNetLMHead(keras.layers.Layer):
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
    def call(self, hidden_states): ...

@keras_serializable
class TFXLNetMainLayer(keras.layers.Layer):
    config_class = XLNetConfig
    def __init__(self, config, **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFSharedEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def build(self, input_shape=...):  # -> None:
        ...
    def create_mask(self, qlen, mlen): ...
    def cache_mem(self, curr_out, prev_mem): ...
    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=...): ...
    def relative_positional_encoding(self, qlen, klen, bsz=...): ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        mems: np.ndarray | tf.Tensor | None = ...,
        perm_mask: np.ndarray | tf.Tensor | None = ...,
        target_mapping: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        input_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any] | tuple[tuple[Any, ...], ...] | list[Any], ...] | TFXLNetModelOutput:
        ...

class TFXLNetPreTrainedModel(TFPreTrainedModel):
    config_class = XLNetConfig
    base_model_prefix = ...

@dataclass
class TFXLNetModelOutput(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    mems: list[tf.Tensor] | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFXLNetLMHeadModelOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    mems: list[tf.Tensor] | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFXLNetForSequenceClassificationOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    mems: list[tf.Tensor] | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFXLNetForTokenClassificationOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    mems: list[tf.Tensor] | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFXLNetForMultipleChoiceOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    mems: list[tf.Tensor] | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFXLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    start_logits: tf.Tensor | None = ...
    end_logits: tf.Tensor | None = ...
    mems: list[tf.Tensor] | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

XLNET_START_DOCSTRING = ...
XLNET_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    XLNET_START_DOCSTRING,
)
class TFXLNetModel(TFXLNetPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        mems: np.ndarray | tf.Tensor | None = ...,
        perm_mask: np.ndarray | tf.Tensor | None = ...,
        target_mapping: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        input_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFXLNetModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    XLNET_START_DOCSTRING,
)
class TFXLNetLMHeadModel(TFXLNetPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_lm_head(self):  # -> TFXLNetLMHead:
        ...
    def get_prefix_bias_name(self): ...
    def prepare_inputs_for_generation(
        self, inputs, past_key_values=..., use_mems=..., **kwargs
    ):  # -> dict[str, Any | None]:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFXLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        mems: np.ndarray | tf.Tensor | None = ...,
        perm_mask: np.ndarray | tf.Tensor | None = ...,
        target_mapping: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        input_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFXLNetLMHeadModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    XLNET_START_DOCSTRING,
)
class TFXLNetForSequenceClassification(
    TFXLNetPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetForSequenceClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        mems: np.ndarray | tf.Tensor | None = ...,
        perm_mask: np.ndarray | tf.Tensor | None = ...,
        target_mapping: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        input_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFXLNetForSequenceClassificationOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    XLNET_START_DOCSTRING,
)
class TFXLNetForMultipleChoice(TFXLNetPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        XLNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetForMultipleChoiceOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        input_mask: np.ndarray | tf.Tensor | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        mems: np.ndarray | tf.Tensor | None = ...,
        perm_mask: np.ndarray | tf.Tensor | None = ...,
        target_mapping: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFXLNetForMultipleChoiceOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    XLNET_START_DOCSTRING,
)
class TFXLNetForTokenClassification(TFXLNetPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetForTokenClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        mems: np.ndarray | tf.Tensor | None = ...,
        perm_mask: np.ndarray | tf.Tensor | None = ...,
        target_mapping: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        input_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFXLNetForTokenClassificationOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    XLNET_START_DOCSTRING,
)
class TFXLNetForQuestionAnsweringSimple(
    TFXLNetPreTrainedModel, TFQuestionAnsweringLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetForQuestionAnsweringSimpleOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        mems: np.ndarray | tf.Tensor | None = ...,
        perm_mask: np.ndarray | tf.Tensor | None = ...,
        target_mapping: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        input_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        start_positions: np.ndarray | tf.Tensor | None = ...,
        end_positions: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFXLNetForQuestionAnsweringSimpleOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFXLNetForMultipleChoice",
    "TFXLNetForQuestionAnsweringSimple",
    "TFXLNetForSequenceClassification",
    "TFXLNetForTokenClassification",
    "TFXLNetLMHeadModel",
    "TFXLNetMainLayer",
    "TFXLNetModel",
    "TFXLNetPreTrainedModel",
]
