from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
from ...configuration_utils import PretrainedConfig
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TFRAG model implementation."""
logger = ...
_CONFIG_FOR_DOC = ...

@dataclass
class TFRetrievAugLMMarginOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    past_key_values: list[tf.Tensor] | None = ...
    doc_scores: tf.Tensor | None = ...
    retrieved_doc_embeds: tf.Tensor | None = ...
    retrieved_doc_ids: tf.Tensor | None = ...
    context_input_ids: tf.Tensor | None = ...
    context_attention_mask: tf.Tensor | None = ...
    question_encoder_last_hidden_state: tf.Tensor | None = ...
    question_enc_hidden_states: tuple[tf.Tensor, ...] | None = ...
    question_enc_attentions: tuple[tf.Tensor, ...] | None = ...
    generator_enc_last_hidden_state: tf.Tensor | None = ...
    generator_enc_hidden_states: tuple[tf.Tensor, ...] | None = ...
    generator_enc_attentions: tuple[tf.Tensor, ...] | None = ...
    generator_dec_hidden_states: tuple[tf.Tensor, ...] | None = ...
    generator_dec_attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFRetrievAugLMOutput(ModelOutput):
    logits: tf.Tensor | None = ...
    past_key_values: list[tf.Tensor] | None = ...
    doc_scores: tf.Tensor | None = ...
    retrieved_doc_embeds: tf.Tensor | None = ...
    retrieved_doc_ids: tf.Tensor | None = ...
    context_input_ids: tf.Tensor | None = ...
    context_attention_mask: tf.Tensor | None = ...
    question_encoder_last_hidden_state: tf.Tensor | None = ...
    question_enc_hidden_states: tuple[tf.Tensor, ...] | None = ...
    question_enc_attentions: tuple[tf.Tensor, ...] | None = ...
    generator_enc_last_hidden_state: tf.Tensor | None = ...
    generator_enc_hidden_states: tuple[tf.Tensor, ...] | None = ...
    generator_enc_attentions: tuple[tf.Tensor, ...] | None = ...
    generator_dec_hidden_states: tuple[tf.Tensor, ...] | None = ...
    generator_dec_attentions: tuple[tf.Tensor, ...] | None = ...

class TFRagPreTrainedModel(TFPreTrainedModel):
    config_class = RagConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str | None = ...,
        generator_pretrained_model_name_or_path: str | None = ...,
        retriever: RagRetriever = ...,
        *model_args,
        **kwargs,
    ) -> TFPreTrainedModel: ...

RAG_START_DOCSTRING = ...
RAG_FORWARD_INPUTS_DOCSTRING = ...

@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class TFRagModel(TFRagPreTrainedModel):
    load_weight_prefix = ...
    def __init__(
        self,
        config: PretrainedConfig | None = ...,
        question_encoder: TFPreTrainedModel | None = ...,
        generator: TFPreTrainedModel | None = ...,
        retriever: RagRetriever | None = ...,
        load_weight_prefix: str | None = ...,
        **kwargs,
    ) -> None: ...
    def set_retriever(self, retriever: RagRetriever):  # -> None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFRetrievAugLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: np.ndarray | tf.Tensor | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        doc_scores: np.ndarray | tf.Tensor | None = ...,
        context_input_ids: np.ndarray | tf.Tensor | None = ...,
        context_attention_mask: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_retrieved: bool | None = ...,
        n_docs: int | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs,
    ) -> TFRetrievAugLMOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings_to_model_forward(
    ...,
    RAG_START_DOCSTRING,
)
class TFRagTokenForGeneration(TFRagPreTrainedModel, TFCausalLanguageModelingLoss):
    load_weight_prefix = ...
    def __init__(
        self,
        config: PretrainedConfig | None = ...,
        question_encoder: TFPreTrainedModel | None = ...,
        generator: TFPreTrainedModel | None = ...,
        retriever: RagRetriever | None = ...,
        **kwargs,
    ) -> None: ...
    def set_retriever(self, retriever: RagRetriever):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=...,
        attention_mask=...,
        use_cache=...,
        encoder_outputs=...,
        doc_scores=...,
        n_docs=...,
        **kwargs,
    ):  # -> dict[str, Any | bool | None]:
        ...
    @property
    def retriever(self):  # -> RagRetriever | None:
        ...
    @property
    def generator(self):  # -> TFPreTrainedModel | None:
        ...
    @property
    def question_encoder(self):  # -> TFPreTrainedModel | None:
        ...
    def marginalize(self, seq_logits, doc_scores, n_docs=...): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFRetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        doc_scores: np.ndarray | tf.Tensor | None = ...,
        context_input_ids: np.ndarray | tf.Tensor | None = ...,
        context_attention_mask: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_retrieved: bool | None = ...,
        n_docs: int | None = ...,
        do_marginalize: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        reduce_loss: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs,
    ) -> TFRetrievAugLMMarginOutput: ...
    def generate(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        context_input_ids=...,
        context_attention_mask=...,
        doc_scores=...,
        n_docs=...,
        generation_config=...,
        logits_processor=...,
        **kwargs,
    ):  # -> TFGreedySearchOutput | TFBeamSearchOutput | TFBeamSampleOutput:
        ...
    def get_input_embeddings(self): ...
    def get_output_embeddings(self):  # -> None:
        ...
    def shift_tokens_right(self, input_ids, start_token_id=...): ...
    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=..., epsilon=..., n_docs=...
    ): ...
    def hf_compute_loss(
        self, labels, y_pred, smooth_epsilon=..., from_logits=..., reduce_loss=...
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings_to_model_forward(
    ...,
    RAG_START_DOCSTRING,
)
class TFRagSequenceForGeneration(TFRagPreTrainedModel, TFCausalLanguageModelingLoss):
    load_weight_prefix = ...
    def __init__(
        self,
        config: PretrainedConfig | None = ...,
        question_encoder: TFPreTrainedModel | None = ...,
        generator: TFPreTrainedModel | None = ...,
        retriever: RagRetriever | None = ...,
        **kwargs,
    ) -> None: ...
    def set_retriever(self, retriever: RagRetriever):  # -> None:
        ...
    @property
    def retriever(self):  # -> RagRetriever | None:
        ...
    @property
    def generator(self):  # -> TFPreTrainedModel | None:
        ...
    @property
    def question_encoder(self):  # -> TFPreTrainedModel | None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFRetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        doc_scores: np.ndarray | tf.Tensor | None = ...,
        context_input_ids: np.ndarray | tf.Tensor | None = ...,
        context_attention_mask: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_retrieved: bool | None = ...,
        n_docs: int | None = ...,
        exclude_bos_score: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        reduce_loss: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs,
    ) -> tuple[tf.Tensor] | TFRetrievAugLMMarginOutput: ...
    def get_nll(
        self,
        seq_logits,
        doc_scores,
        target,
        reduce_loss=...,
        epsilon=...,
        exclude_bos_score=...,
        n_docs=...,
    ): ...
    def generate(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        context_input_ids=...,
        context_attention_mask=...,
        doc_scores=...,
        do_deduplication=...,
        num_return_sequences=...,
        num_beams=...,
        n_docs=...,
        **model_kwargs,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFRagModel",
    "TFRagPreTrainedModel",
    "TFRagSequenceForGeneration",
    "TFRagTokenForGeneration",
]
