import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType

"""
Feature extractor class for Whisper
"""
logger = ...

class WhisperFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size=...,
        sampling_rate=...,
        hop_length=...,
        chunk_length=...,
        n_fft=...,
        padding_value=...,
        dither=...,
        return_attention_mask=...,
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
        truncation: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_attention_mask: bool | None = ...,
        padding: str | None = ...,
        max_length: int | None = ...,
        sampling_rate: int | None = ...,
        do_normalize: bool | None = ...,
        device: str | None = ...,
        return_token_timestamps: bool | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["WhisperFeatureExtractor"]
