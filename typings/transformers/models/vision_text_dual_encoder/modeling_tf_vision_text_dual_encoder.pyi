import tensorflow as tf

from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig
from ..clip.modeling_tf_clip import TFCLIPOutput
from ...modeling_tf_utils import TFPreTrainedModel, unpack_inputs
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TensorFlow VisionTextDualEncoder model."""
logger = ...
_CONFIG_FOR_DOC = ...
VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = ...
VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING = ...
VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING = ...
VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = ...

def contrastive_loss(logits: tf.Tensor) -> tf.Tensor: ...
def clip_loss(similarity: tf.Tensor) -> tf.Tensor: ...

@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class TFVisionTextDualEncoderModel(TFPreTrainedModel):
    config_class = VisionTextDualEncoderConfig
    base_model_prefix = ...
    load_weight_prefix = ...
    def __init__(
        self,
        config: VisionTextDualEncoderConfig | None = ...,
        vision_model: TFPreTrainedModel | None = ...,
        text_model: TFPreTrainedModel | None = ...,
    ) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def tf_to_pt_weight_rename(self, tf_weight):  # -> tuple[str] | tuple[Any]:
        ...
    @add_start_docstrings_to_model_forward(
        VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING
    )
    def get_text_features(
        self,
        input_ids=...,
        attention_mask=...,
        position_ids=...,
        token_type_ids=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ): ...
    @add_start_docstrings_to_model_forward(
        VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING
    )
    def get_image_features(
        self,
        pixel_values=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCLIPOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        return_loss: bool | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFCLIPOutput: ...
    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str | None = ...,
        text_model_name_or_path: str | None = ...,
        *model_args,
        **kwargs,
    ) -> TFPreTrainedModel: ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any]:
        ...

__all__ = ["TFVisionTextDualEncoderModel"]
