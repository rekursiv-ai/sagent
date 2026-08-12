from typing import Any

from torch import nn

import torch

from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
)
from ..llama.tokenization_llama import LlamaTokenizer
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import TextInput
from ...utils import TransformersKwargs

VOCAB_FILES_NAMES = ...
SPIECE_UNDERLINE = ...
logger = ...

class GemmaConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        head_dim=...,
        hidden_act=...,
        hidden_activation=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

class GemmaTokenizer(LlamaTokenizer, PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        sp_model_kwargs: dict[str, Any] | None = ...,
        add_bos_token=...,
        add_eos_token=...,
        clean_up_tokenization_spaces=...,
        use_default_system_prompt=...,
        spaces_between_special_tokens=...,
        **kwargs,
    ) -> None: ...
    def get_spm_processor(self): ...
    def unk_token_length(self): ...
    def tokenize(self, text: TextInput, **kwargs) -> list[str]: ...
    def convert_tokens_to_string(self, tokens): ...

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = ...) -> None: ...
    def forward(self, x): ...
    def extra_repr(self):  # -> str:
        ...

class GemmaMLP(LlamaMLP):
    def __init__(self, config) -> None: ...

class GemmaRotaryEmbedding(LlamaRotaryEmbedding): ...
class GemmaPreTrainedModel(LlamaPreTrainedModel): ...

class GemmaModel(LlamaModel):
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPast: ...

class GemmaForCausalLM(LlamaForCausalLM):
    def forward(**super_kwargs):  # -> CausalLMOutputWithPast:
        ...

class GemmaForSequenceClassification(LlamaForSequenceClassification): ...
class GemmaForTokenClassification(LlamaForTokenClassification): ...

__all__ = [
    "GemmaConfig",
    "GemmaForCausalLM",
    "GemmaForSequenceClassification",
    "GemmaForTokenClassification",
    "GemmaModel",
    "GemmaPreTrainedModel",
    "GemmaTokenizer",
]
