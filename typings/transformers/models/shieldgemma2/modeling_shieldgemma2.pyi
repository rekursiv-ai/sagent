from dataclasses import dataclass

import torch

from .configuration_shieldgemma2 import ShieldGemma2Config
from ...cache_utils import Cache
from ...modeling_outputs import ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

logger = ...

@dataclass
class ShieldGemma2ImageClassifierOutputWithNoAttention(
    ImageClassifierOutputWithNoAttention
):
    probabilities: torch.Tensor | None = ...

@auto_docstring
class ShieldGemma2ForImageClassification(PreTrainedModel):
    config: ShieldGemma2Config
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: ShieldGemma2Config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Any:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def tie_weights(self):  # -> Any:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **lm_kwargs,
    ) -> ShieldGemma2ImageClassifierOutputWithNoAttention: ...

__all__ = ["ShieldGemma2ForImageClassification"]
