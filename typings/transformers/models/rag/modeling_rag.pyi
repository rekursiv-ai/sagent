from collections.abc import Callable
from dataclasses import dataclass

import torch

from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""RAG model implementation."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class RetrievAugLMMarginOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    doc_scores: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    retrieved_doc_embeds: torch.FloatTensor | None = ...
    retrieved_doc_ids: torch.LongTensor | None = ...
    context_input_ids: torch.LongTensor | None = ...
    context_attention_mask: torch.LongTensor | None = ...
    question_encoder_last_hidden_state: torch.FloatTensor | None = ...
    question_enc_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    question_enc_attentions: tuple[torch.FloatTensor, ...] | None = ...
    generator_enc_last_hidden_state: torch.FloatTensor | None = ...
    generator_enc_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    generator_enc_attentions: tuple[torch.FloatTensor, ...] | None = ...
    generator_dec_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    generator_dec_attentions: tuple[torch.FloatTensor, ...] | None = ...
    generator_cross_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring
class RetrievAugLMOutput(ModelOutput):
    logits: torch.FloatTensor | None = ...
    doc_scores: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    retrieved_doc_embeds: torch.FloatTensor | None = ...
    retrieved_doc_ids: torch.LongTensor | None = ...
    context_input_ids: torch.LongTensor | None = ...
    context_attention_mask: torch.LongTensor | None = ...
    question_encoder_last_hidden_state: torch.FloatTensor | None = ...
    question_enc_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    question_enc_attentions: tuple[torch.FloatTensor, ...] | None = ...
    generator_enc_last_hidden_state: torch.FloatTensor | None = ...
    generator_enc_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    generator_enc_attentions: tuple[torch.FloatTensor, ...] | None = ...
    generator_dec_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    generator_dec_attentions: tuple[torch.FloatTensor, ...] | None = ...
    generator_cross_attentions: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring(custom_intro=...)
@auto_docstring
class RagPreTrainedModel(PreTrainedModel):
    config: RagConfig
    base_model_prefix = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str | None = ...,
        generator_pretrained_model_name_or_path: str | None = ...,
        retriever: RagRetriever = ...,
        **kwargs,
    ) -> PreTrainedModel: ...

@auto_docstring
class RagModel(RagPreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig | None = ...,
        question_encoder: PreTrainedModel | None = ...,
        generator: PreTrainedModel | None = ...,
        retriever: RagRetriever | None = ...,
        **kwargs,
    ) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        past_key_values: Cache | None = ...,
        doc_scores: torch.FloatTensor | None = ...,
        context_input_ids: torch.LongTensor | None = ...,
        context_attention_mask: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_retrieved: bool | None = ...,
        n_docs: int | None = ...,
    ) -> tuple[torch.Tensor] | RetrievAugLMOutput: ...

@auto_docstring(custom_intro=...)
class RagSequenceForGeneration(RagPreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig | None = ...,
        question_encoder: PreTrainedModel | None = ...,
        generator: PreTrainedModel | None = ...,
        retriever: RagRetriever | None = ...,
        **kwargs,
    ) -> None: ...
    def set_retriever(self, retriever: RagRetriever):  # -> None:
        ...
    def set_context_encoder_for_training(
        self, ctx_encoder: PreTrainedModel
    ):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        past_key_values: Cache | None = ...,
        context_input_ids: torch.LongTensor | None = ...,
        context_attention_mask: torch.LongTensor | None = ...,
        doc_scores: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_retrieved: bool | None = ...,
        exclude_bos_score: bool | None = ...,
        reduce_loss: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        n_docs: int | None = ...,
        **kwargs,
    ) -> RetrievAugLMMarginOutput: ...
    @property
    def retriever(self):  # -> RagRetriever | None:
        ...
    @property
    def generator(self):  # -> PreTrainedModel | None:
        ...
    @property
    def question_encoder(self):  # -> PreTrainedModel | None:
        ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        context_input_ids: torch.LongTensor | None = ...,
        context_attention_mask: torch.LongTensor | None = ...,
        doc_scores: torch.FloatTensor | None = ...,
        do_deduplication: bool | None = ...,
        num_return_sequences: int | None = ...,
        num_beams: int | None = ...,
        n_docs: int | None = ...,
        **model_kwargs,
    ) -> torch.LongTensor: ...
    def get_nll(
        self,
        seq_logits,
        doc_scores,
        target,
        reduce_loss=...,
        epsilon=...,
        exclude_bos_score=...,
        n_docs=...,
    ):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class RagTokenForGeneration(RagPreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config: PretrainedConfig | None = ...,
        question_encoder: PreTrainedModel | None = ...,
        generator: PreTrainedModel | None = ...,
        retriever: RagRetriever | None = ...,
        **kwargs,
    ) -> None: ...
    def set_retriever(self, retriever: RagRetriever):  # -> None:
        ...
    def set_context_encoder_for_training(
        self, ctx_encoder: PreTrainedModel
    ):  # -> None:
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
    def generator(self):  # -> PreTrainedModel | None:
        ...
    @property
    def question_encoder(self):  # -> PreTrainedModel | None:
        ...
    def marginalize(self, seq_logits, doc_scores, n_docs=...):  # -> Tensor:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        past_key_values: Cache | None = ...,
        context_input_ids: torch.LongTensor | None = ...,
        context_attention_mask: torch.LongTensor | None = ...,
        doc_scores: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_retrieved: bool | None = ...,
        do_marginalize: bool | None = ...,
        reduce_loss: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        n_docs: int | None = ...,
        **kwargs,
    ) -> RetrievAugLMMarginOutput: ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        context_input_ids: torch.LongTensor | None = ...,
        context_attention_mask: torch.LongTensor | None = ...,
        doc_scores: torch.FloatTensor | None = ...,
        n_docs: int | None = ...,
        generation_config: GenerationConfig | None = ...,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
        stopping_criteria: StoppingCriteriaList | None = ...,
        **kwargs,
    ) -> torch.LongTensor: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def get_output_embeddings(self):  # -> None:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def shift_tokens_right(self, input_ids, start_token_id=...): ...
    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=..., epsilon=..., n_docs=...
    ):  # -> Tensor:
        ...

__all__ = [
    "RagModel",
    "RagPreTrainedModel",
    "RagSequenceForGeneration",
    "RagTokenForGeneration",
]
