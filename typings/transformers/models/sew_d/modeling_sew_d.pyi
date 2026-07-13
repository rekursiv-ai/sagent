from torch import nn

import torch

from .configuration_sew_d import SEWDConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch SEW model."""
logger = ...
_HIDDEN_STATES_START_POSITION = ...

def make_log_bucket_position(relative_pos, bucket_size, max_position):  # -> Tensor:
    ...
def build_relative_position(
    query_size, key_size, bucket_size=..., max_position=..., device=...
):  # -> Tensor:
    ...
@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos): ...
@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer): ...
@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer): ...
def get_mask(input, local_context):  # -> tuple[Tensor | None, Any | int]:
    ...

class SEWDNoLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class SEWDLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class SEWDGroupNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class SEWDPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class SEWDSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None: ...
    def forward(self, hidden_states): ...

class SEWDUpsampling(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class SEWDFeatureEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_values):  # -> Any:
        ...

class SEWDFeatureExtractor(SEWDFeatureEncoder):
    def __init__(self, config) -> None: ...

class ContextPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...
    @property
    def output_dim(self): ...

class XSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, dim):  # -> Tensor:
        ...
    @staticmethod
    def backward(ctx, grad_output):  # -> tuple[Tensor, None, None]:
        ...
    @staticmethod
    def symbolic(g, self, mask, dim):  # -> Value | tuple[Value, ...]:
        ...

class DropoutContext:
    def __init__(self) -> None: ...

class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, local_ctx): ...
    @staticmethod
    def backward(ctx, grad_output):  # -> tuple[Any, None]:
        ...
    @staticmethod
    def symbolic(
        g: torch._C.Graph,
        input: torch._C.Value,
        local_ctx: float | DropoutContext,
    ) -> torch._C.Value: ...

class StableDropout(nn.Module):
    def __init__(self, drop_prob) -> None: ...
    def forward(self, x):  # -> Any | None:
        ...
    def clear_context(self):  # -> None:
        ...
    def init_context(self, reuse_mask=..., scale=...):  # -> None:
        ...
    def get_context(self):  # -> Any:
        ...

class SEWDSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

class DisentangledSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def transpose_for_scores(self, x, attention_heads): ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=...,
        query_states=...,
        relative_pos=...,
        rel_embeddings=...,
    ):  # -> tuple[Tensor, Any] | Tensor:
        ...
    def disentangled_attention_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):  # -> Tensor | Literal[0]:
        ...

class SEWDAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=...,
        query_states=...,
        relative_pos=...,
        rel_embeddings=...,
    ):  # -> tuple[Any, Any] | Any:
        ...

class SEWDIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class SEWDOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

class SEWDLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=...,
        relative_pos=...,
        rel_embeddings=...,
        output_attentions=...,
    ):  # -> tuple[Any, Any] | Any:
        ...

class ConvLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, residual_states, input_mask):  # -> Any:
        ...

class SEWDTransformerEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def get_rel_embedding(self):  # -> Any | Tensor | None:
        ...
    def get_attention_mask(self, attention_mask): ...
    def get_rel_pos(
        self, hidden_states, query_states=..., relative_pos=...
    ):  # -> Tensor | None:
        ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=...,
        output_attentions=...,
        query_states=...,
        relative_pos=...,
        return_dict=...,
    ):  # -> tuple[object | Any | tuple[object | Any, ...] | tuple[()] | tuple[Any, ...], ...] | BaseModelOutput:
        ...

class SEWDEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, ...] | BaseModelOutput:
        ...

@auto_docstring
class SEWDPreTrainedModel(PreTrainedModel):
    config: SEWDConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

@auto_docstring
class SEWDModel(SEWDPreTrainedModel):
    def __init__(self, config: SEWDConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        mask_time_indices: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class SEWDForCTC(SEWDPreTrainedModel):
    def __init__(self, config, target_lang: str | None = ...) -> None: ...
    def tie_weights(self):  # -> None:
        ...
    def freeze_feature_extractor(self):  # -> None:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    def freeze_base_model(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | CausalLMOutput: ...

@auto_docstring(custom_intro=...)
class SEWDForSequenceClassification(SEWDPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_feature_extractor(self):  # -> None:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    def freeze_base_model(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

__all__ = [
    "SEWDForCTC",
    "SEWDForSequenceClassification",
    "SEWDModel",
    "SEWDPreTrainedModel",
]
