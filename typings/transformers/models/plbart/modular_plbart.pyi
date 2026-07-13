from torch import nn

import torch

from .configuration_plbart import PLBartConfig
from ..bart.modeling_bart import (
    BartClassificationHead,
    BartDecoder,
    BartEncoder,
    BartForCausalLM,
    BartScaledWordEmbedding,
)
from ..bigbird_pegasus.modeling_bigbird_pegasus import (
    BigBirdPegasusForSequenceClassification,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch PLBART model."""

class PLBartScaledWordEmbedding(BartScaledWordEmbedding): ...

@auto_docstring
class PLBartPreTrainedModel(PreTrainedModel):
    config: PLBartConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

class PLBartEncoder(BartEncoder): ...
class PLBartDecoder(BartDecoder): ...

@auto_docstring
class PLBartModel(PLBartPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: PLBartConfig) -> None: ...
    def get_input_embeddings(self):  # -> PLBartScaledWordEmbedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> PLBartEncoder:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.LongTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class PLBartForConditionalGeneration(PLBartPreTrainedModel, GenerationMixin):
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _tied_weights_keys = ...
    def __init__(self, config: PLBartConfig) -> None: ...
    def get_encoder(self):  # -> PLBartEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | PLBartModel:
        ...
    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: int | None = ...,
        mean_resizing: bool = ...,
    ) -> nn.Embedding: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.LongTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

class PLBartClassificationHead(BartClassificationHead): ...

class PLBartForSequenceClassification(BigBirdPegasusForSequenceClassification):
    def forward(**super_kwargs):  # -> None:
        ...

class PLBartForCausalLM(BartForCausalLM):
    @auto_docstring
    def forward(**super_kwargs):  # -> None:
        ...

__all__ = [
    "PLBartForCausalLM",
    "PLBartForConditionalGeneration",
    "PLBartForSequenceClassification",
    "PLBartModel",
    "PLBartPreTrainedModel",
]
