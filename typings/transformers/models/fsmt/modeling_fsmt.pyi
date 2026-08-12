from typing import Any

from torch import Tensor, nn

import torch

from .configuration_fsmt import FSMTConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch Fairseq model, ported from https://github.com/pytorch/fairseq/tree/master/examples/wmt19"""
logger = ...

def invert_mask(attention_mask): ...
def triu_onnx(x, diagonal=...): ...

@auto_docstring
class PretrainedFSMTModel(PreTrainedModel):
    config: FSMTConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any | Tensor]:
        ...

def shift_tokens_right(input_ids, pad_token_id): ...
def make_padding_mask(input_ids, padding_idx=...):  # -> None:
    ...

class EncoderLayer(nn.Module):
    def __init__(self, config: FSMTConfig) -> None: ...
    def forward(
        self, x, encoder_padding_mask, layer_head_mask, output_attentions=...
    ):  # -> tuple[Any, Any]:
        ...

class FSMTEncoder(nn.Module):
    def __init__(self, config: FSMTConfig, embed_tokens) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Tensor | Any | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class DecoderLayer(nn.Module):
    def __init__(self, config: FSMTConfig, layer_idx=...) -> None: ...
    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=...,
        layer_state=...,
        causal_mask=...,
        layer_head_mask=...,
        cross_attn_layer_head_mask=...,
        decoder_padding_mask=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Any, Any]:
        ...

class FSMTDecoder(nn.Module):
    def __init__(self, config: FSMTConfig, embed_tokens: nn.Embedding) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
        decoder_causal_mask: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Any | EncoderDecoderCache | Cache | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any, ...], ...] | BaseModelOutputWithPastAndCrossAttentions:
        ...

class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=...,
        bias=...,
        encoder_decoder_attention=...,
        layer_idx=...,
    ) -> None: ...
    def forward(
        self,
        query,
        key: Tensor | None,
        key_padding_mask: Tensor | None = ...,
        layer_state: Cache | None = ...,
        attn_mask: Tensor | None = ...,
        layer_head_mask: Tensor | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[Tensor, Tensor | None]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor | None]: ...

def fill_with_neg_inf(t): ...

@auto_docstring
class FSMTModel(PretrainedFSMTModel):
    _tied_weights_keys = ...
    def __init__(self, config: FSMTConfig) -> None: ...
    def get_encoder(self):  # -> FSMTEncoder:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqModelOutput: ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Embedding:
        ...
    def set_output_embeddings(self, value):  # -> None:
        ...

@auto_docstring(custom_intro=...)
class FSMTForConditionalGeneration(PretrainedFSMTModel, GenerationMixin):
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config: FSMTConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...
    def get_encoder(self):  # -> FSMTEncoder:
        ...
    def get_decoder(self):  # -> FSMTDecoder:
        ...
    def get_output_embeddings(self):  # -> Embedding:
        ...
    def set_output_embeddings(self, value):  # -> None:
        ...

class SinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions, embedding_dim, padding_idx) -> None: ...
    def make_weight(self, num_positions, embedding_dim, padding_idx):  # -> None:
        ...
    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):  # -> Tensor:
        ...
    @staticmethod
    def make_positions(tensor, padding_idx: int): ...
    def forward(
        self,
        input,
        incremental_state: Any | None = ...,
        timestep: Tensor | None = ...,
    ):  # -> Tensor:
        ...

__all__ = ["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"]
