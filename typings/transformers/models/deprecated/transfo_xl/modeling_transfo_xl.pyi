from dataclasses import dataclass

from torch import nn

import torch

from .configuration_transfo_xl import TransfoXLConfig
from ....modeling_utils import PreTrainedModel
from ....utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""
PyTorch Transformer XL model. Adapted from https://github.com/kimiyoung/transformer-xl. In particular
https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

def build_tf_to_pytorch_map(model, config):  # -> dict[Any, Any]:
    ...
def load_tf_weights_in_transfo_xl(model, config, tf_path): ...

class PositionalEmbedding(nn.Module):
    def __init__(self, demb) -> None: ...
    def forward(self, pos_seq, bsz=...):  # -> Tensor:
        ...

class PositionwiseFF(nn.Module):
    def __init__(
        self, d_model, d_inner, dropout, pre_lnorm=..., layer_norm_epsilon=...
    ) -> None: ...
    def forward(self, inp):  # -> Any:
        ...

class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=...,
        pre_lnorm=...,
        r_r_bias=...,
        r_w_bias=...,
        layer_norm_epsilon=...,
    ) -> None: ...
    def forward(
        self, w, r, attn_mask=..., mems=..., head_mask=..., output_attentions=...
    ):  # -> list[Any]:
        ...

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        layer_norm_epsilon=...,
        **kwargs,
    ) -> None: ...
    def forward(
        self,
        dec_inp,
        r,
        dec_attn_mask=...,
        mems=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> Any:
        ...

class AdaptiveEmbedding(nn.Module):
    def __init__(
        self, n_token, d_embed, d_proj, cutoffs, div_val=..., sample_softmax=...
    ) -> None: ...
    def forward(self, inp):  # -> Tensor | Any:
        ...

class TransfoXLPreTrainedModel(PreTrainedModel):
    config: TransfoXLConfig
    load_tf_weights = ...
    base_model_prefix = ...
    def resize_token_embeddings(
        self, new_num_tokens: int | None = ..., layer: int | None = ...
    ):  # -> Module | Any:
        ...

@dataclass
class TransfoXLModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
class TransfoXLSequenceClassifierOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
class TransfoXLLMHeadModelOutput(ModelOutput):
    losses: torch.FloatTensor | None = ...
    prediction_scores: torch.FloatTensor | None = ...
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    loss: torch.FloatTensor | None = ...
    @property
    def logits(self):  # -> FloatTensor | None:
        ...

TRANSFO_XL_START_DOCSTRING = ...
TRANSFO_XL_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    TRANSFO_XL_START_DOCSTRING,
)
class TransfoXLModel(TransfoXLPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> AdaptiveEmbedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def backward_compatible(self):  # -> None:
        ...
    def reset_memory_length(self, mem_len):  # -> None:
        ...
    def init_mems(self, bsz):  # -> list[Any] | None:
        ...
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TransfoXLModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        mems: list[torch.FloatTensor] | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TransfoXLModelOutput: ...

@add_start_docstrings(
    ...,
    TRANSFO_XL_START_DOCSTRING,
)
class TransfoXLLMHeadModel(TransfoXLPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def tie_weights(self):  # -> None:
        ...
    def reset_memory_length(self, mem_len):  # -> None:
        ...
    def init_mems(self, bsz):  # -> list[Any] | None:
        ...
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TransfoXLLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        mems: list[torch.FloatTensor] | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TransfoXLLMHeadModelOutput: ...
    def get_output_embeddings(self):  # -> Tensor | Module:
        ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=..., **model_kwargs
    ):  # -> dict[Any, Any]:
        ...

@add_start_docstrings(
    ...,
    TRANSFO_XL_START_DOCSTRING,
)
class TransfoXLForSequenceClassification(TransfoXLPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TransfoXLSequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        mems: list[torch.FloatTensor] | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TransfoXLSequenceClassifierOutputWithPast: ...

__all__ = [
    "AdaptiveEmbedding",
    "TransfoXLForSequenceClassification",
    "TransfoXLLMHeadModel",
    "TransfoXLModel",
    "TransfoXLPreTrainedModel",
    "load_tf_weights_in_transfo_xl",
]
