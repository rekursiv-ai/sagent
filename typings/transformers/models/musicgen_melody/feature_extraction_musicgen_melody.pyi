from typing import Any

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType
from ...utils.import_utils import requires

"""
Feature extractor class for Musicgen Melody
"""
logger = ...

@requires(backends=("torchaudio",))
class MusicgenMelodyFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size=...,
        sampling_rate=...,
        hop_length=...,
        chunk_length=...,
        n_fft=...,
        num_chroma=...,
        padding_value=...,
        return_attention_mask=...,
        stem_indices=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_attention_mask: bool | None = ...,
        padding: str | None = ...,
        max_length: int | None = ...,
        sampling_rate: int | None = ...,
        **kwargs,
    ) -> BatchFeature: ...
    def to_dict(self) -> dict[str, Any]: ...

__all__ = ["MusicgenMelodyFeatureExtractor"]
