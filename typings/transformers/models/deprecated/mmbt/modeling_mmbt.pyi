from torch import nn

from ....modeling_outputs import BaseModelOutputWithPooling
from ....modeling_utils import ModuleUtilsMixin
from ....utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""PyTorch MMBT model."""
logger = ...
_CONFIG_FOR_DOC = ...

class ModalEmbeddings(nn.Module):
    def __init__(self, config, encoder, embeddings) -> None: ...
    def forward(
        self,
        input_modal,
        start_token=...,
        end_token=...,
        position_ids=...,
        token_type_ids=...,
    ):  # -> Any:
        ...

MMBT_START_DOCSTRING = ...
MMBT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    MMBT_START_DOCSTRING,
)
class MMBTModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, transformer, encoder) -> None: ...
    @add_start_docstrings_to_model_forward(MMBT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_modal,
        input_ids=...,
        modal_start_tokens=...,
        modal_end_tokens=...,
        attention_mask=...,
        token_type_ids=...,
        modal_token_type_ids=...,
        position_ids=...,
        modal_position_ids=...,
        head_mask=...,
        inputs_embeds=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> BaseModelOutputWithPooling:
        ...
    def get_input_embeddings(self):  # -> Tensor | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MMBT_START_DOCSTRING,
    MMBT_INPUTS_DOCSTRING,
)
class MMBTForClassification(nn.Module):
    def __init__(self, config, transformer, encoder) -> None: ...
    def forward(
        self,
        input_modal,
        input_ids=...,
        modal_start_tokens=...,
        modal_end_tokens=...,
        attention_mask=...,
        token_type_ids=...,
        modal_token_type_ids=...,
        position_ids=...,
        modal_position_ids=...,
        head_mask=...,
        inputs_embeds=...,
        labels=...,
        return_dict=...,
    ):  # -> SequenceClassifierOutput:
        ...

__all__ = ["MMBTForClassification", "MMBTModel", "ModalEmbeddings"]
