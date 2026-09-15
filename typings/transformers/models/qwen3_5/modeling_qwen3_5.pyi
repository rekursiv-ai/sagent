from typing import override

from torch import Tensor, nn

from .configuration_qwen3_5 import Qwen3_5TextConfig
from ...cache_utils import Cache

class Qwen3_5CausalLMOutputWithPast:
    logits: Tensor
    past_key_values: Cache | None

class Qwen3_5TextModel(nn.Module):
    config: Qwen3_5TextConfig
    layers: nn.ModuleList

    def __init__(self, config: Qwen3_5TextConfig) -> None: ...

class Qwen3_5ForCausalLM(nn.Module):
    config: Qwen3_5TextConfig
    model: Qwen3_5TextModel
    lm_head: nn.Linear

    def __init__(self, config: Qwen3_5TextConfig) -> None: ...
    @override
    def forward(
        self,
        input_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        position_ids: Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: Tensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: object,
    ) -> Qwen3_5CausalLMOutputWithPast: ...

class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None: ...
    @override
    def forward(
        self,
        hidden_states: Tensor,
        cache_params: Cache | None = ...,
        attention_mask: Tensor | None = ...,
        **kwargs: object,
    ) -> Tensor: ...

class Qwen3_5VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None: ...
