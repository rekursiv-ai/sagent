from dataclasses import dataclass

from torch import nn

import torch

from .configuration_decision_transformer import DecisionTransformerConfig
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch DecisionTransformer model."""
logger = ...

def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path): ...
def eager_attention_forward(
    module, query, key, value, attention_mask, head_mask=..., **kwargs
):  # -> tuple[Tensor, Any]:
    ...

class DecisionTransformerGPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=..., layer_idx=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]: ...

class DecisionTransformerGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None: ...
    def forward(
        self, hidden_states: tuple[torch.FloatTensor] | None
    ) -> torch.FloatTensor: ...

class DecisionTransformerGPT2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> (
        tuple[torch.Tensor] | tuple[torch.Tensor, tuple[torch.FloatTensor, ...]] | None
    ): ...

@auto_docstring
class DecisionTransformerGPT2PreTrainedModel(PreTrainedModel):
    config: DecisionTransformerConfig
    load_tf_weights = ...
    base_model_prefix = ...
    is_parallelizable = ...
    supports_gradient_checkpointing = ...
    _can_compile_fullgraph = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

class DecisionTransformerGPT2Model(DecisionTransformerGPT2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@dataclass
@auto_docstring(custom_intro=...)
class DecisionTransformerOutput(ModelOutput):
    state_preds: torch.FloatTensor | None = ...
    action_preds: torch.FloatTensor | None = ...
    return_preds: torch.FloatTensor | None = ...
    hidden_states: torch.FloatTensor | None = ...
    attentions: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...

class DecisionTransformerPreTrainedModel(PreTrainedModel):
    config: DecisionTransformerConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

@auto_docstring(
    custom_intro="""
    The Decision Transformer Model
    """
)
class DecisionTransformerModel(DecisionTransformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        states: torch.FloatTensor | None = ...,
        actions: torch.FloatTensor | None = ...,
        rewards: torch.FloatTensor | None = ...,
        returns_to_go: torch.FloatTensor | None = ...,
        timesteps: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | DecisionTransformerOutput: ...

__all__ = [
    "DecisionTransformerGPT2Model",
    "DecisionTransformerGPT2PreTrainedModel",
    "DecisionTransformerModel",
    "DecisionTransformerPreTrainedModel",
]
