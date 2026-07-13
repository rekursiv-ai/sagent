import torch

from .configuration_retribert import RetriBertConfig
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings

"""
RetriBERT model
"""
logger = ...

class RetriBertPreTrainedModel(PreTrainedModel):
    config: RetriBertConfig
    load_tf_weights = ...
    base_model_prefix = ...

RETRIBERT_START_DOCSTRING = ...

@add_start_docstrings(
    ...,
    RETRIBERT_START_DOCSTRING,
)
class RetriBertModel(RetriBertPreTrainedModel):
    def __init__(self, config: RetriBertConfig) -> None: ...
    def embed_sentences_checkpointed(
        self, input_ids, attention_mask, sent_encoder, checkpoint_batch_size=...
    ):  # -> Tensor:
        ...
    def embed_questions(
        self, input_ids, attention_mask=..., checkpoint_batch_size=...
    ):  # -> Any:
        ...
    def embed_answers(
        self, input_ids, attention_mask=..., checkpoint_batch_size=...
    ):  # -> Any:
        ...
    def forward(
        self,
        input_ids_query: torch.LongTensor,
        attention_mask_query: torch.FloatTensor | None,
        input_ids_doc: torch.LongTensor,
        attention_mask_doc: torch.FloatTensor | None,
        checkpoint_batch_size: int = ...,
    ) -> torch.FloatTensor: ...

__all__ = ["RetriBertModel", "RetriBertPreTrainedModel"]
