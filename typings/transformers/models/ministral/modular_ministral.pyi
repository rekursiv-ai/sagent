import torch

from ..mistral.configuration_mistral import MistralConfig
from ..qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2ForQuestionAnswering,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2MLP,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs

class MinistralConfig(MistralConfig, PretrainedConfig):
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
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        sliding_window=...,
        attention_dropout=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

class MinistralMLP(Qwen2MLP): ...

class MinistralAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: int) -> None: ...

class MinistralRMSNorm(Qwen2RMSNorm): ...
class MinistralDecoderLayer(Qwen2DecoderLayer): ...
class MinistralPreTrainedModel(Qwen2PreTrainedModel): ...
class MinistralRotaryEmbedding(Qwen2RotaryEmbedding): ...

class MinistralModel(Qwen2Model):
    def __init__(self, config: MinistralConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
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

class MinistralForCausalLM(Qwen2ForCausalLM): ...
class MinistralForSequenceClassification(Qwen2ForSequenceClassification): ...
class MinistralForTokenClassification(Qwen2ForTokenClassification): ...
class MinistralForQuestionAnswering(Qwen2ForQuestionAnswering): ...

__all__ = [
    "MinistralConfig",
    "MinistralForCausalLM",
    "MinistralForQuestionAnswering",
    "MinistralForSequenceClassification",
    "MinistralForTokenClassification",
    "MinistralModel",
    "MinistralPreTrainedModel",
]
