from typing import Any

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType

"""Feature extractor class for SpeechT5."""
logger = ...

class SpeechT5FeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size: int = ...,
        sampling_rate: int = ...,
        padding_value: float = ...,
        do_normalize: bool = ...,
        num_mel_bins: int = ...,
        hop_length: int = ...,
        win_length: int = ...,
        win_function: str = ...,
        frame_signal_scale: float = ...,
        fmin: float = ...,
        fmax: float = ...,
        mel_floor: float = ...,
        reduction_factor: int = ...,
        return_attention_mask: bool = ...,
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
        audio: np.ndarray
        | list[float]
        | list[np.ndarray]
        | list[list[float]]
        | None = ...,
        audio_target: np.ndarray
        | list[float]
        | list[np.ndarray]
        | list[list[float]]
        | None = ...,
        padding: bool | str | PaddingStrategy = ...,
        max_length: int | None = ...,
        truncation: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_attention_mask: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        sampling_rate: int | None = ...,
        **kwargs,
    ) -> BatchFeature: ...
    def to_dict(self) -> dict[str, Any]: ...

__all__ = ["SpeechT5FeatureExtractor"]
