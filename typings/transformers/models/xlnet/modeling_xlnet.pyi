from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_xlnet import XLNetConfig
from ...generation import GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""
PyTorch XLNet model.
"""
logger = ...

def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=...):  # -> dict[Any, Any]:
    ...
def load_tf_weights_in_xlnet(model, config, tf_path): ...

class XLNetRelativeAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def prune_heads(self, heads): ...
    @staticmethod
    def rel_shift(x, klen=...):  # -> Tensor:
        ...
    @staticmethod
    def rel_shift_bnij(x, klen=...):  # -> Tensor:
        ...
    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        seg_mat=...,
        attn_mask=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> tuple[Tensor, Tensor] | Tensor:
        ...
    def post_attention(self, h, attn_vec, residual=...):  # -> Any:
        ...
    def forward(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=...,
        target_mapping=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> tuple[Any, Any | None, tuple[Tensor | Any, Tensor | Any] | Any | Tensor] | tuple[Any, Any | None]:
        ...

class XLNetFeedForward(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, inp):  # -> Any:
        ...

class XLNetLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        output_h,
        output_g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=...,
        target_mapping=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> Any:
        ...
    def ff_chunk(self, output_x):  # -> Any:
        ...

class XLNetPoolerStartLogits(nn.Module):
    def __init__(self, config: XLNetConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        p_mask: torch.FloatTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class XLNetPoolerEndLogits(nn.Module):
    def __init__(self, config: XLNetConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        p_mask: torch.FloatTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class XLNetPoolerAnswerClass(nn.Module):
    def __init__(self, config: XLNetConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        cls_index: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class XLNetSequenceSummary(nn.Module):
    def __init__(self, config: XLNetConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        cls_index: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

@auto_docstring
class XLNetPreTrainedModel(PreTrainedModel):
    config: XLNetConfig
    load_tf_weights = ...
    base_model_prefix = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`XLNetModel`].
    """
)
class XLNetModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`XLNetLMHeadModel`].
    """
)
class XLNetLMHeadModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class XLNetForSequenceClassificationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class XLNetForTokenClassificationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class XLNetForMultipleChoiceOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class XLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_logits: torch.FloatTensor | None = ...
    end_logits: torch.FloatTensor | None = ...
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class XLNetForQuestionAnsweringOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_top_log_probs: torch.FloatTensor | None = ...
    start_top_index: torch.LongTensor | None = ...
    end_top_log_probs: torch.FloatTensor | None = ...
    end_top_index: torch.LongTensor | None = ...
    cls_logits: torch.FloatTensor | None = ...
    mems: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring
class XLNetModel(XLNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def create_mask(self, qlen, mlen):  # -> Tensor:
        ...
    def cache_mem(self, curr_out, prev_mem):  # -> Tensor:
        ...
    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=...):  # -> Tensor:
        ...
    def relative_positional_encoding(self, qlen, klen, bsz=...):  # -> Tensor:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        mems: torch.Tensor | None = ...,
        perm_mask: torch.Tensor | None = ...,
        target_mapping: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        input_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | XLNetModelOutput: ...

@auto_docstring(custom_intro=...)
class XLNetLMHeadModel(XLNetPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=..., use_mems=..., **kwargs
    ):  # -> dict[str, Tensor | Any | None]:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        mems: torch.Tensor | None = ...,
        perm_mask: torch.Tensor | None = ...,
        target_mapping: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        input_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | XLNetLMHeadModelOutput: ...

@auto_docstring(custom_intro=...)
class XLNetForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        mems: torch.Tensor | None = ...,
        perm_mask: torch.Tensor | None = ...,
        target_mapping: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        input_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | XLNetForSequenceClassificationOutput: ...

@auto_docstring
class XLNetForTokenClassification(XLNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        mems: torch.Tensor | None = ...,
        perm_mask: torch.Tensor | None = ...,
        target_mapping: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        input_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | XLNetForTokenClassificationOutput: ...

@auto_docstring
class XLNetForMultipleChoice(XLNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        input_mask: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        mems: torch.Tensor | None = ...,
        perm_mask: torch.Tensor | None = ...,
        target_mapping: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | XLNetForMultipleChoiceOutput: ...

@auto_docstring(custom_intro=...)
class XLNetForQuestionAnsweringSimple(XLNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        mems: torch.Tensor | None = ...,
        perm_mask: torch.Tensor | None = ...,
        target_mapping: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        input_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | XLNetForQuestionAnsweringSimpleOutput: ...

@auto_docstring
class XLNetForQuestionAnswering(XLNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        mems: torch.Tensor | None = ...,
        perm_mask: torch.Tensor | None = ...,
        target_mapping: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        input_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        is_impossible: torch.Tensor | None = ...,
        cls_index: torch.Tensor | None = ...,
        p_mask: torch.Tensor | None = ...,
        use_mems: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | XLNetForQuestionAnsweringOutput: ...

__all__ = [
    "XLNetForMultipleChoice",
    "XLNetForQuestionAnswering",
    "XLNetForQuestionAnsweringSimple",
    "XLNetForSequenceClassification",
    "XLNetForTokenClassification",
    "XLNetLMHeadModel",
    "XLNetModel",
    "XLNetPreTrainedModel",
    "load_tf_weights_in_xlnet",
]
