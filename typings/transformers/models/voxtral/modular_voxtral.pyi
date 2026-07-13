from torch import nn

import torch

from .configuration_voxtral import VoxtralConfig
from ..qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioAttention,
    Qwen2AudioEncoder,
    Qwen2AudioEncoderLayer,
    Qwen2AudioPreTrainedModel,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs

class VoxtralAttention(Qwen2AudioAttention): ...
class VoxtralEncoderLayer(Qwen2AudioEncoderLayer): ...

class VoxtralPreTrainedModel(Qwen2AudioPreTrainedModel):
    _supports_flex_attn = ...
    _supports_cache_class = ...
    _supports_attention_backend = ...
    _can_compile_fullgraph = ...
    _no_split_modules = ...

@auto_docstring(custom_intro=...)
class VoxtralEncoder(Qwen2AudioEncoder):
    _can_record_outputs = ...
    @check_model_inputs
    def forward(
        self, input_features, attention_mask=..., **kwargs: Unpack[TransformersKwargs]
    ):  # -> BaseModelOutput:
        ...

class VoxtralMultiModalProjector(nn.Module):
    def __init__(self, config: VoxtralConfig) -> None: ...
    def forward(self, audio_features):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class VoxtralForConditionalGeneration(VoxtralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    _keep_in_fp32_modules_strict = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Any:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_audio_features(self, input_features: torch.FloatTensor):  # -> Any:
        ...
    def get_audio_embeds(self, input_features: torch.FloatTensor):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(self, *args, **kwargs):  # -> dict[Any, Any]:
        ...

__all__ = [
    "VoxtralEncoder",
    "VoxtralForConditionalGeneration",
    "VoxtralPreTrainedModel",
]
