import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType

"""
Feature extractor class for Speech2Text
"""
logger = ...

class Speech2TextFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size=...,
        sampling_rate=...,
        num_mel_bins=...,
        padding_value=...,
        dither=...,
        do_ceptral_normalize=...,
        normalize_means=...,
        normalize_vars=...,
        **kwargs,
    ) -> None: ...
    @staticmethod
    def utterance_cmvn(
        x: np.ndarray,
        input_length: int,
        normalize_means: bool | None = ...,
        normalize_vars: bool | None = ...,
        padding_value: float = ...,
    ) -> np.ndarray: ...
    def normalize(
        self,
        input_features: list[np.ndarray],
        attention_mask: np.ndarray | None = ...,
    ) -> list[np.ndarray]: ...
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy = ...,
        max_length: int | None = ...,
        truncation: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        sampling_rate: int | None = ...,
        return_attention_mask: bool | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["Speech2TextFeatureExtractor"]
