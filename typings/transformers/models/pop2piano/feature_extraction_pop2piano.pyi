import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import (
    TensorType,
)
from ...utils.import_utils import requires

"""Feature extractor class for Pop2Piano"""
logger = ...

@requires(backends=("essentia", "librosa", "scipy", "torch"))
class Pop2PianoFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        sampling_rate: int = ...,
        padding_value: int = ...,
        window_size: int = ...,
        hop_length: int = ...,
        min_frequency: float = ...,
        feature_size: int = ...,
        num_bars: int = ...,
        **kwargs,
    ) -> None: ...
    def mel_spectrogram(self, sequence: np.ndarray):  # -> NDArray[Any]:
        ...
    def extract_rhythm(self, audio: np.ndarray):  # -> tuple[Any, Any, Any, Any, Any]:
        ...
    def interpolate_beat_times(
        self,
        beat_times: np.ndarray,
        steps_per_beat: np.ndarray,
        n_extend: np.ndarray,
    ): ...
    def preprocess_mel(
        self, audio: np.ndarray, beatstep: np.ndarray
    ):  # -> tuple[NDArray[Any], Any]:
        ...
    def pad(
        self,
        inputs: BatchFeature,
        is_batched: bool,
        return_attention_mask: bool,
        return_tensors: str | TensorType | None = ...,
    ):  # -> BatchFeature:
        ...
    def __call__(
        self,
        audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | list[int],
        steps_per_beat: int = ...,
        resample: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["Pop2PianoFeatureExtractor"]
