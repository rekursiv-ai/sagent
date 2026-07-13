from typing import Literal

from flash_attn.layers.rotary import RotaryEmbedding
from torch import nn

import torch

from ..gemma.modeling_gemma import GemmaRotaryEmbedding
from ...configuration_utils import PretrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, is_flash_attn_2_available

if is_flash_attn_2_available(): ...
else:
    RotaryEmbedding = ...
logger = ...

class ModernBertConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        hidden_activation=...,
        max_position_embeddings=...,
        initializer_range=...,
        initializer_cutoff_factor=...,
        norm_eps=...,
        norm_bias=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        cls_token_id=...,
        sep_token_id=...,
        global_rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        global_attn_every_n_layers=...,
        local_attention=...,
        local_rope_theta=...,
        embedding_dropout=...,
        mlp_bias=...,
        mlp_dropout=...,
        decoder_bias=...,
        classifier_pooling: Literal["cls", "mean"] = ...,
        classifier_dropout=...,
        classifier_bias=...,
        classifier_activation=...,
        deterministic_flash_attn=...,
        sparse_prediction=...,
        sparse_pred_ignore_index=...,
        reference_compile=...,
        repad_logits_with_grad=...,
        **kwargs,
    ) -> None: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cu_seqlens: torch.Tensor | None = ...,
        max_seqlen: int | None = ...,
    ): ...
    @staticmethod
    def backward(ctx, do):  # -> tuple[Any, None, None, None, None, None, None]:
        ...

def apply_rotary_unpadded(
    qkv,
    cos,
    sin,
    cu_seqlens: torch.Tensor | None = ...,
    max_seqlen: int | None = ...,
):  # -> Any | None:
    ...

class ModernBertUnpaddedRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        base: float = ...,
        max_seqlen: int | None = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> None: ...
    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = ...,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def extra_repr(self) -> str: ...

class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig) -> None: ...
    @torch.compile(dynamic=True)
    def compiled_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class ModernBertRotaryEmbedding(GemmaRotaryEmbedding): ...

def eager_attention_forward(
    module: ModernBertAttention,
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: torch.LongTensor | None,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: bool | None = ...,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...
def flash_attention_forward(
    module: ModernBertAttention,
    qkv: torch.Tensor,
    rotary_emb: ModernBertUnpaddedRotaryEmbedding,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = ...,
    **_kwargs,
) -> tuple[torch.Tensor]: ...
def sdpa_attention_forward(
    module: ModernBertAttention,
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: torch.LongTensor | None,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    **_kwargs,
) -> tuple[torch.Tensor]: ...

MODERNBERT_ATTENTION_FUNCTION = ...

class ModernBertAttention(nn.Module):
    def __init__(
        self, config: ModernBertConfig, layer_id: int | None = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class ModernBertEncoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: ModernBertConfig, layer_id: int | None = ...
    ) -> None: ...
    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        sliding_window_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cu_seqlens: torch.Tensor | None = ...,
        max_seqlen: int | None = ...,
        output_attentions: bool | None = ...,
    ) -> torch.Tensor: ...

@auto_docstring
class ModernBertPreTrainedModel(PreTrainedModel):
    config: ModernBertConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    def resize_token_embeddings(self, *args, **kwargs):  # -> Embedding:
        ...

@auto_docstring
class ModernBertModel(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        sliding_window_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        indices: torch.Tensor | None = ...,
        cu_seqlens: torch.Tensor | None = ...,
        max_seqlen: int | None = ...,
        batch_size: int | None = ...,
        seq_len: int | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutput: ...

class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class ModernBertForMaskedLM(ModernBertPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: ModernBertConfig) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings: nn.Linear):  # -> None:
        ...
    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        sliding_window_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        indices: torch.Tensor | None = ...,
        cu_seqlens: torch.Tensor | None = ...,
        max_seqlen: int | None = ...,
        batch_size: int | None = ...,
        seq_len: int | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | MaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class ModernBertForSequenceClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        sliding_window_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        indices: torch.Tensor | None = ...,
        cu_seqlens: torch.Tensor | None = ...,
        max_seqlen: int | None = ...,
        batch_size: int | None = ...,
        seq_len: int | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

@auto_docstring(custom_intro=...)
class ModernBertForTokenClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        sliding_window_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        indices: torch.Tensor | None = ...,
        cu_seqlens: torch.Tensor | None = ...,
        max_seqlen: int | None = ...,
        batch_size: int | None = ...,
        seq_len: int | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput: ...

@auto_docstring
class ModernBertForQuestionAnswering(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        sliding_window_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        indices: torch.Tensor | None = ...,
        cu_seqlens: torch.Tensor | None = ...,
        max_seqlen: int | None = ...,
        batch_size: int | None = ...,
        seq_len: int | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput: ...

@auto_docstring(custom_intro=...)
class ModernBertForMultipleChoice(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        sliding_window_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        indices: torch.Tensor | None = ...,
        cu_seqlens: torch.Tensor | None = ...,
        max_seqlen: int | None = ...,
        batch_size: int | None = ...,
        seq_len: int | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | MultipleChoiceModelOutput: ...

__all__ = [
    "ModernBertConfig",
    "ModernBertForMaskedLM",
    "ModernBertForMultipleChoice",
    "ModernBertForQuestionAnswering",
    "ModernBertForSequenceClassification",
    "ModernBertForTokenClassification",
    "ModernBertModel",
    "ModernBertPreTrainedModel",
]
