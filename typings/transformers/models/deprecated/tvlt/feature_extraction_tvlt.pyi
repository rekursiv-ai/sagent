import numpy as np

from ....feature_extraction_sequence_utils import BatchFeature, SequenceFeatureExtractor
from ....utils import TensorType

"""Feature extractor class for TVLT."""
logger = ...

class TvltFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        spectrogram_length=...,
        num_channels=...,
        patch_size=...,
        feature_size=...,
        sampling_rate=...,
        hop_length_to_sampling_rate=...,
        n_fft=...,
        padding_value=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        return_tensors: str | TensorType | None = ...,
        return_attention_mask: bool | None = ...,
        sampling_rate: int | None = ...,
        resample: bool = ...,
        mask_audio: bool = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["TvltFeatureExtractor"]
