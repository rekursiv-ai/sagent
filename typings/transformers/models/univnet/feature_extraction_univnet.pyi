from typing import Any

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType

"""Feature extractor class for UnivNetModel."""
logger = ...

class UnivNetFeatureExtractor(SequenceFeatureExtractor):
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
        filter_length: int | None = ...,
        max_length_s: int = ...,
        fmin: float = ...,
        fmax: float | None = ...,
        mel_floor: float = ...,
        center: bool = ...,
        compression_factor: float = ...,
        compression_clip_val: float = ...,
        normalize_min: float = ...,
        normalize_max: float = ...,
        model_in_channels: int = ...,
        pad_end_length: int = ...,
        return_attention_mask=...,
        **kwargs,
    ) -> None: ...
    def normalize(self, spectrogram): ...
    def denormalize(self, spectrogram): ...
    def mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray: ...
    def generate_noise(
        self, noise_length: int, generator: np.random.Generator | None = ...
    ) -> np.ndarray: ...
    def batch_decode(self, waveforms, waveform_lengths=...) -> list[np.ndarray]: ...
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | None = ...,
        padding: bool | str | PaddingStrategy = ...,
        max_length: int | None = ...,
        truncation: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_noise: bool = ...,
        generator: np.random.Generator | None = ...,
        pad_end: bool = ...,
        pad_length: int | None = ...,
        do_normalize: str | None = ...,
        return_attention_mask: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
    ) -> BatchFeature: ...
    def to_dict(self) -> dict[str, Any]: ...

__all__ = ["UnivNetFeatureExtractor"]
