from dataclasses import dataclass

from torch import Tensor

import torch

from .configuration_timm_wrapper import TimmWrapperConfig
from ...modeling_outputs import ImageClassifierOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

@dataclass
@auto_docstring(custom_intro=...)
class TimmWrapperModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring
class TimmWrapperPreTrainedModel(PreTrainedModel):
    main_input_name = ...
    config: TimmWrapperConfig
    _no_split_modules = ...
    model_tags = ...
    accepts_loss_kwargs = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def post_init(self):  # -> None:
        ...
    def load_state_dict(self, state_dict, *args, **kwargs):  # -> _IncompatibleKeys:
        ...

class TimmWrapperModel(TimmWrapperPreTrainedModel):
    def __init__(self, config: TimmWrapperConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | list[int] | None = ...,
        return_dict: bool | None = ...,
        do_pooling: bool | None = ...,
        **kwargs,
    ) -> TimmWrapperModelOutput | tuple[Tensor, ...]: ...

class TimmWrapperForImageClassification(TimmWrapperPreTrainedModel):
    def __init__(self, config: TimmWrapperConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | list[int] | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> ImageClassifierOutput | tuple[Tensor, ...]: ...

__all__ = [
    "TimmWrapperForImageClassification",
    "TimmWrapperModel",
    "TimmWrapperPreTrainedModel",
]
