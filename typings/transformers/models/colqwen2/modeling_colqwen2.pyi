from dataclasses import dataclass

from torch import nn

import torch

from .configuration_colqwen2 import ColQwen2Config
from ...cache_utils import Cache
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple

@auto_docstring
class ColQwen2PreTrainedModel(PreTrainedModel):
    config: ColQwen2Config
    base_model_prefix = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...

@dataclass
@auto_docstring(custom_intro=...)
class ColQwen2ForRetrievalOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    embeddings: torch.Tensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@auto_docstring(custom_intro=...)
class ColQwen2ForRetrieval(ColQwen2PreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: ColQwen2Config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        labels: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        pixel_values: torch.Tensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> ColQwen2ForRetrievalOutput: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Any:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def tie_weights(self):  # -> Any:
        ...
    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        mean_resizing: bool = ...,
    ) -> nn.Embedding: ...

__all__ = ["ColQwen2ForRetrieval", "ColQwen2PreTrainedModel"]
