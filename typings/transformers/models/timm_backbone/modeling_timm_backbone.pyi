from torch import Tensor

import torch

from .configuration_timm_backbone import TimmBackboneConfig
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils.backbone_utils import BackboneMixin

class TimmBackbone(PreTrainedModel, BackboneMixin):
    main_input_name = ...
    supports_gradient_checkpointing = ...
    config: TimmBackboneConfig
    def __init__(self, config, **kwargs) -> None: ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, **kwargs
    ):  # -> Self:
        ...
    def freeze_batch_norm_2d(self):  # -> None:
        ...
    def unfreeze_batch_norm_2d(self):  # -> None:
        ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> BackboneOutput | tuple[Tensor, ...]: ...

__all__ = ["TimmBackbone"]
