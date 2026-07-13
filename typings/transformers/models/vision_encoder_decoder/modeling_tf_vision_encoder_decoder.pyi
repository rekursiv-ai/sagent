import numpy as np
import tensorflow as tf

from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig
from ...configuration_utils import PretrainedConfig
from ...modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqLMOutput
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    unpack_inputs,
)
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""Classes to support TF Vision-Encoder-Text-Decoder architectures"""
logger = ...
_CONFIG_FOR_DOC = ...
DEPRECATION_WARNING = ...
VISION_ENCODER_DECODER_START_DOCSTRING = ...
VISION_ENCODER_DECODER_INPUTS_DOCSTRING = ...

def shift_tokens_right(
    input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int
): ...

@add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING)
class TFVisionEncoderDecoderModel(TFPreTrainedModel, TFCausalLanguageModelingLoss):
    config_class = VisionEncoderDecoderConfig
    base_model_prefix = ...
    load_weight_prefix = ...
    main_input_name = ...
    def __init__(
        self,
        config: PretrainedConfig | None = ...,
        encoder: TFPreTrainedModel | None = ...,
        decoder: TFPreTrainedModel | None = ...,
    ) -> None: ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...
    def get_encoder(self):  # -> TFPreTrainedModel | None:
        ...
    def get_input_embeddings(self): ...
    def get_output_embeddings(self):  # -> None:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def tf_to_pt_weight_rename(self, tf_weight):  # -> tuple[str] | tuple[Any]:
        ...
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str | None = ...,
        decoder_pretrained_model_name_or_path: str | None = ...,
        *model_args,
        **kwargs,
    ) -> TFPreTrainedModel: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        VISION_ENCODER_DECODER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        pixel_values: np.ndarray | tf.Tensor | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: tuple | TFBaseModelOutput | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs,
    ) -> TFSeq2SeqLMOutput | tuple[tf.Tensor]: ...
    def serving_output(self, output):  # -> TFSeq2SeqLMOutput:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        use_cache=...,
        encoder_outputs=...,
        **kwargs,
    ):  # -> dict[str, Any | TFBaseModelOutput | None]:
        ...
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor): ...
    def resize_token_embeddings(self, *args, **kwargs): ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFVisionEncoderDecoderModel"]
