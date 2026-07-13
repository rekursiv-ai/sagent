import torch

from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig
from ..clip.modeling_clip import CLIPOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, filter_out_non_signature_kwargs

"""PyTorch VisionTextDualEncoder model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def clip_loss(similarity: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class VisionTextDualEncoderModel(PreTrainedModel):
    config: VisionTextDualEncoderConfig
    base_model_prefix = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    def __init__(
        self,
        config: VisionTextDualEncoderConfig | None = ...,
        vision_model: PreTrainedModel | None = ...,
        text_model: PreTrainedModel | None = ...,
    ) -> None: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.FloatTensor: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | CLIPOutput: ...
    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str | None = ...,
        text_model_name_or_path: str | None = ...,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel: ...

__all__ = ["VisionTextDualEncoderModel"]
