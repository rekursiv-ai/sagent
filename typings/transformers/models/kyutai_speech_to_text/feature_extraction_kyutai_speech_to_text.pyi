import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType

logger = ...

class KyutaiSpeechToTextFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size: int = ...,
        sampling_rate: int = ...,
        padding_value: float = ...,
        chunk_length_s: float | None = ...,
        overlap: float | None = ...,
        audio_delay_seconds: float | None = ...,
        audio_silence_prefix_seconds: float | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def chunk_length(self) -> int | None: ...
    @property
    def chunk_stride(self) -> int | None: ...
    def __call__(
        self,
        raw_audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy | None = ...,
        truncation: bool | None = ...,
        max_length: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        sampling_rate: int | None = ...,
    ) -> BatchFeature: ...

__all__ = ["KyutaiSpeechToTextFeatureExtractor"]
