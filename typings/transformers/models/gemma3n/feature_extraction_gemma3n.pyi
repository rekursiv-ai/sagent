from collections.abc import Sequence

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType

logger = ...

def create_fb_matrix(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    fft_length: int,
    norm: str | None = ...,
) -> np.ndarray: ...

class Gemma3nAudioFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size: int = ...,
        sampling_rate: int = ...,
        padding_value: float = ...,
        return_attention_mask: bool = ...,
        frame_length_ms: float = ...,
        hop_length_ms: float = ...,
        min_frequency: float = ...,
        max_frequency: float = ...,
        preemphasis: float = ...,
        preemphasis_htk_flavor: bool = ...,
        fft_overdrive: bool = ...,
        dither: float = ...,
        input_scale_factor: float = ...,
        mel_floor: float = ...,
        per_bin_mean: Sequence[float] | None = ...,
        per_bin_stddev: Sequence[float] | None = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy = ...,
        max_length: int | None = ...,
        truncation: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_attention_mask: bool | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["Gemma3nAudioFeatureExtractor"]
