from typing import Any
from torch import nn

import torch

from .configuration_gptsan_japanese import GPTSanJapaneseConfig
from ....cache_utils import Cache
from ....modeling_outputs import (
    MoECausalLMOutputWithPast,
    MoEModelOutputWithPastAndCrossAttentions,
)
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ....utils.deprecation import deprecate_kwarg

"""PyTorch GPTSANJapanese model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...

def router_z_loss_func(router_logits: torch.Tensor) -> float: ...
def load_balancing_loss_func(
    router_probs: torch.Tensor, expert_indices: torch.Tensor
) -> float: ...

class GPTSanJapaneseDenseActDense(nn.Module):
    def __init__(self, config: GPTSanJapaneseConfig, ext_layer=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class GPTSanJapaneseTop1Router(nn.Module):
    def __init__(self, config: GPTSanJapaneseConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> tuple: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple: ...

class GPTSanJapaneseSparseMLP(nn.Module):
    def __init__(
        self, config: GPTSanJapaneseConfig, expert_class: nn.Module = ...
    ) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, tuple[Any, Tensor]]:
        ...

class GPTSanJapaneseLayerSparseFF(nn.Module):
    def __init__(self, config: GPTSanJapaneseConfig) -> None: ...
    def forward(self, hidden_states, output_router_logits):  # -> tuple[Any, Any]:
        ...

class GPTSanJapaneseLayerDenseFF(nn.Module):
    def __init__(self, config: GPTSanJapaneseConfig) -> None: ...
    def forward(self, hidden_states): ...

class GPTSanJapaneseAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: GPTSanJapaneseConfig | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class GPTSanJapaneseLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]: ...

class GPTSanJapaneseBlock(nn.Module):
    def __init__(self, config, ext_layer=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_router_tuple: bool | None = ...,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]: ...

class GPTSanJapanesePreTrainedModel(PreTrainedModel):
    config: GPTSanJapaneseConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Tensor]:
        ...

GPTSAN_JAPANESE_START_DOCSTRING = ...
GPTSAN_JAPANESE_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    GPTSAN_JAPANESE_START_DOCSTRING,
)
class GPTSanJapaneseModel(GPTSanJapanesePreTrainedModel):
    def __init__(self, config: GPTSanJapaneseConfig) -> None: ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.FloatTensor | None = ...,
        spout: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        output_router_logits: bool | None = ...,
        num_precontext: torch.LongTensor | None = ...,
    ) -> MoEModelOutputWithPastAndCrossAttentions | tuple[torch.FloatTensor]: ...

@add_start_docstrings(
    ...,
    GPTSAN_JAPANESE_START_DOCSTRING,
)
class GPTSanJapaneseForConditionalGeneration(GPTSanJapanesePreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: GPTSanJapaneseConfig) -> None: ...
    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.FloatTensor | None = ...,
        spout: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        output_router_logits: bool | None = ...,
        labels: torch.LongTensor | None = ...,
    ) -> tuple[torch.FloatTensor] | MoECausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        token_type_ids: torch.FloatTensor | None = ...,
        spout: list | torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        **kwargs,
    ):  # -> dict[str, Tensor | FloatTensor | list[Any] | Cache | None] | dict[str, LongTensor | FloatTensor | list[Any] | None]:
        ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...
    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: int | None = ...,
        mean_resizing: bool = ...,
    ) -> nn.Embedding: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...

__all__ = [
    "GPTSanJapaneseForConditionalGeneration",
    "GPTSanJapaneseModel",
    "GPTSanJapanesePreTrainedModel",
]
