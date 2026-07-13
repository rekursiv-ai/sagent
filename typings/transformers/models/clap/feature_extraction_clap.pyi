from typing import Any

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType
from ...utils.import_utils import requires

"""Feature extractor class for CLAP."""
logger = ...

@requires(backends=("torch",))
class ClapFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size=...,
        sampling_rate=...,
        hop_length=...,
        max_length_s=...,
        fft_window_size=...,
        padding_value=...,
        return_attention_mask=...,
        frequency_min: float = ...,
        frequency_max: float = ...,
        top_db: int | None = ...,
        truncation: str = ...,
        padding: str = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: str | None = ...,
        padding: str | None = ...,
        max_length: int | None = ...,
        sampling_rate: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["ClapFeatureExtractor"]
