import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType

"""
Feature extractor class for CLVP
"""
logger = ...

class ClvpFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size=...,
        sampling_rate=...,
        default_audio_length=...,
        hop_length=...,
        chunk_length=...,
        n_fft=...,
        padding_value=...,
        mel_norms=...,
        return_attention_mask=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | None = ...,
        truncation: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_attention_mask: bool | None = ...,
        padding: str | None = ...,
        max_length: int | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["ClvpFeatureExtractor"]
