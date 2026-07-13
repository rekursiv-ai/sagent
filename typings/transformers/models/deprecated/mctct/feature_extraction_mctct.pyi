import numpy as np

from ....feature_extraction_sequence_utils import SequenceFeatureExtractor
from ....feature_extraction_utils import BatchFeature
from ....file_utils import PaddingStrategy, TensorType

"""
Feature extractor class for M-CTC-T
"""
logger = ...

class MCTCTFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size=...,
        sampling_rate=...,
        padding_value=...,
        hop_length=...,
        win_length=...,
        win_function=...,
        frame_signal_scale=...,
        preemphasis_coeff=...,
        mel_floor=...,
        normalize_means=...,
        normalize_vars=...,
        return_attention_mask=...,
        **kwargs,
    ) -> None: ...
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
        return_attention_mask: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        sampling_rate: int | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["MCTCTFeatureExtractor"]
