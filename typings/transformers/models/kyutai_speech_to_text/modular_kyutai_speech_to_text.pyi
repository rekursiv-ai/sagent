from torch import nn

import numpy as np
import torch

from ..encodec.feature_extraction_encodec import EncodecFeatureExtractor
from ..llama.modeling_llama import LlamaForCausalLM
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from ..moshi.modeling_moshi import MoshiModel, MoshiPreTrainedModel
from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...utils import PaddingStrategy, TensorType

logger = ...

class KyutaiSpeechToTextFeatureExtractor(EncodecFeatureExtractor):
    def __init__(
        self,
        audio_delay_seconds: float | None = ...,
        audio_silence_prefix_seconds: float | None = ...,
        **super_kwargs,
    ) -> None: ...
    def __call__(
        self,
        raw_audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy | None = ...,
        truncation: bool | None = ...,
        max_length: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        sampling_rate: int | None = ...,
    ) -> BatchFeature: ...

class KyutaiSpeechToTextPreTrainedModel(MoshiPreTrainedModel): ...
class KyutaiSpeechToTextConv1dPaddingCache(MimiConv1dPaddingCache): ...

class KyutaiSpeechToTextEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_ids):  # -> Any:
        ...

class KyutaiSpeechToTextModel(MoshiModel):
    def __init__(self, config) -> None: ...

class KyutaiSpeechToTextForConditionalGeneration(LlamaForCausalLM, GenerationMixin):
    _keep_in_fp32_modules_strict = ...
    def __init__(self, config) -> None: ...
    def forward(self, **super_kwargs):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        *args,
        audio_tokens: torch.LongTensor | None = ...,
        input_values: torch.FloatTensor | None = ...,
        padding_mask: torch.Tensor | None = ...,
        audio_window_size: int | None = ...,
        current_window: tuple[int, int] | None = ...,
        encoder_past_key_values: Cache | None = ...,
        padding_cache: KyutaiSpeechToTextConv1dPaddingCache | None = ...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    @classmethod
    def from_pretrained(
        cls, *args, **kwargs
    ):  # -> tuple[Any | PreTrainedModel, Any] | PreTrainedModel:
        ...
    def save_pretrained(self, *args, **kwargs):  # -> None:
        ...
    def generate(self, *args, **kwargs):  # -> GenerateOutput | LongTensor:
        ...

__all__ = [
    "KyutaiSpeechToTextFeatureExtractor",
    "KyutaiSpeechToTextForConditionalGeneration",
    "KyutaiSpeechToTextModel",
    "KyutaiSpeechToTextPreTrainedModel",
]
