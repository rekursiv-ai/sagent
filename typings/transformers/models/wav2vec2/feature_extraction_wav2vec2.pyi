import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType

"""
Feature extractor class for Wav2Vec2
"""
logger = ...

class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size=...,
        sampling_rate=...,
        padding_value=...,
        return_attention_mask=...,
        do_normalize=...,
        **kwargs,
    ) -> None: ...
    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: list[np.ndarray],
        attention_mask: list[np.ndarray],
        padding_value: float = ...,
    ) -> list[np.ndarray]: ...
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy = ...,
        max_length: int | None = ...,
        truncation: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_attention_mask: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        sampling_rate: int | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["Wav2Vec2FeatureExtractor"]
