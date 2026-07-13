import torch

from .configuration_biogpt import BioGptConfig
from ..bart.modeling_bart import (
    BartAttention,
    BartDecoderLayer,
    BartScaledWordEmbedding,
)
from ..opt.modeling_opt import OPTLearnedPositionalEmbedding
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch BioGPT model."""

class BioGptLearnedPositionalEmbedding(OPTLearnedPositionalEmbedding):
    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = ...,
        position_ids: torch.LongTensor | None = ...,
    ):  # -> None:
        ...

class BioGptScaledWordEmbedding(BartScaledWordEmbedding): ...
class BioGptAttention(BartAttention): ...

class BioGptDecoderLayer(BartDecoderLayer):
    def __init__(self, config: BioGptConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class BioGptPreTrainedModel(PreTrainedModel):
    config: BioGptConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...

@auto_docstring
class BioGptModel(BioGptPreTrainedModel):
    def __init__(self, config: BioGptConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class BioGptForCausalLM(BioGptPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...

@auto_docstring
class BioGptForTokenClassification(BioGptPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring(custom_intro=...)
class BioGptForSequenceClassification(BioGptPreTrainedModel):
    def __init__(self, config: BioGptConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
    ) -> tuple | SequenceClassifierOutputWithPast: ...
    def get_input_embeddings(self):  # -> BioGptScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...

__all__ = [
    "BioGptForCausalLM",
    "BioGptForSequenceClassification",
    "BioGptForTokenClassification",
    "BioGptModel",
    "BioGptPreTrainedModel",
]
