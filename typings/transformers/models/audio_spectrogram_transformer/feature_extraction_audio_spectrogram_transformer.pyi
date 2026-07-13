import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType

"""
Feature extractor class for Audio Spectrogram Transformer.
"""
logger = ...

class ASTFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size=...,
        sampling_rate=...,
        num_mel_bins=...,
        max_length=...,
        padding_value=...,
        do_normalize=...,
        mean=...,
        std=...,
        return_attention_mask=...,
        **kwargs,
    ) -> None: ...
    def normalize(self, input_values: np.ndarray) -> np.ndarray: ...
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["ASTFeatureExtractor"]
