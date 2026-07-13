import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType
from ...utils.import_utils import requires

EPSILON = ...
LOG_ZERO_GUARD_VALUE = ...
logger = ...

@requires(backends=("torch", "librosa"))
class ParakeetFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size=...,
        sampling_rate=...,
        hop_length=...,
        n_fft=...,
        win_length=...,
        preemphasis=...,
        padding_value=...,
        **kwargs,
    ) -> None: ...
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

__all__ = ["ParakeetFeatureExtractor"]
