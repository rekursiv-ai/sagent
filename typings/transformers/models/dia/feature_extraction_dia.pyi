import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType

"""Feature extractor class for Dia"""
logger = ...

class DiaFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size: int = ...,
        sampling_rate: int = ...,
        padding_value: float = ...,
        hop_length: int = ...,
        **kwargs,
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

__all__ = ["DiaFeatureExtractor"]
