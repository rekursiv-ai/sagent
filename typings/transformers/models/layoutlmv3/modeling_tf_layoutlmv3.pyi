import tensorflow as tf

from .configuration_layoutlmv3 import LayoutLMv3Config
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 LayoutLMv3 model."""
_CONFIG_FOR_DOC = ...
_DUMMY_INPUT_IDS = ...
_DUMMY_BBOX = ...
LARGE_NEGATIVE = ...

class TFLayoutLMv3PatchEmbeddings(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMv3TextEmbeddings(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def calculate_spatial_position_embeddings(self, bbox: tf.Tensor) -> tf.Tensor: ...
    def create_position_ids_from_inputs_embeds(
        self, inputs_embds: tf.Tensor
    ) -> tf.Tensor: ...
    def create_position_ids_from_input_ids(self, input_ids: tf.Tensor) -> tf.Tensor: ...
    def create_position_ids(
        self, input_ids: tf.Tensor, inputs_embeds: tf.Tensor
    ) -> tf.Tensor: ...
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        bbox: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMv3SelfAttention(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def transpose_for_scores(self, x: tf.Tensor): ...
    def cogview_attention(self, attention_scores: tf.Tensor, alpha: float = ...): ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None,
        head_mask: tf.Tensor | None,
        output_attentions: bool,
        rel_pos: tf.Tensor | None = ...,
        rel_2d_pos: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | tuple[tf.Tensor, tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMv3SelfOutput(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMv3Attention(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None,
        head_mask: tf.Tensor | None,
        output_attentions: bool,
        rel_pos: tf.Tensor | None = ...,
        rel_2d_pos: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | tuple[tf.Tensor, tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMv3Intermediate(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMv3Output(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMv3Layer(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None,
        head_mask: tf.Tensor | None,
        output_attentions: bool,
        rel_pos: tf.Tensor | None = ...,
        rel_2d_pos: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | tuple[tf.Tensor, tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMv3Encoder(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def relative_position_bucket(
        self, relative_positions: tf.Tensor, num_buckets: int, max_distance: int
    ): ...
    def call(
        self,
        hidden_states: tf.Tensor,
        bbox: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        position_ids: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> (
        TFBaseModelOutput
        | tuple[tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFLayoutLMv3MainLayer(keras.layers.Layer):
    config_class = LayoutLMv3Config
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    def set_input_embeddings(self, value: tf.Variable):  # -> None:
        ...
    def init_visual_bbox(
        self, image_size: tuple[int, int], max_len: int = ...
    ):  # -> None:
        ...
    def calculate_visual_bbox(self, batch_size: int, dtype: tf.DType): ...
    def embed_image(self, pixel_values: tf.Tensor) -> tf.Tensor: ...
    def get_extended_attention_mask(self, attention_mask: tf.Tensor) -> tf.Tensor: ...
    def get_head_mask(
        self, head_mask: tf.Tensor | None
    ) -> tf.Tensor | list[tf.Tensor | None]: ...
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        bbox: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> (
        TFBaseModelOutput
        | tuple[tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ): ...

class TFLayoutLMv3PreTrainedModel(TFPreTrainedModel):
    config_class = LayoutLMv3Config
    base_model_prefix = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

LAYOUTLMV3_START_DOCSTRING = ...
LAYOUTLMV3_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3Model(TFLayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        bbox: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> (
        TFBaseModelOutput
        | tuple[tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMv3ClassificationHead(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    def call(self, inputs: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForSequenceClassification(
    TFLayoutLMv3PreTrainedModel, TFSequenceClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        bbox: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> (
        TFSequenceClassifierOutput
        | tuple[tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForTokenClassification(
    TFLayoutLMv3PreTrainedModel, TFTokenClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        bbox: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        pixel_values: tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> (
        TFTokenClassifierOutput
        | tuple[tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForQuestionAnswering(
    TFLayoutLMv3PreTrainedModel, TFQuestionAnsweringLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: LayoutLMv3Config, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        start_positions: tf.Tensor | None = ...,
        end_positions: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        bbox: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> (
        TFQuestionAnsweringModelOutput
        | tuple[tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFLayoutLMv3ForQuestionAnswering",
    "TFLayoutLMv3ForSequenceClassification",
    "TFLayoutLMv3ForTokenClassification",
    "TFLayoutLMv3Model",
    "TFLayoutLMv3PreTrainedModel",
]
