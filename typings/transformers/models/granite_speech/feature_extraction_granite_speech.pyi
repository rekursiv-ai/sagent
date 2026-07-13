from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...tokenization_utils_base import AudioInput

"""Feature extractor class for Granite Speech."""
logger = ...

class GraniteSpeechFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ...
    def __init__(
        self,
        sampling_rate: int = ...,
        n_fft: int = ...,
        win_length: int = ...,
        hop_length: int = ...,
        n_mels: int = ...,
        projector_window_size: int = ...,
        projector_downsample_rate: int = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self, audios: AudioInput, device: str | None = ...
    ) -> BatchFeature: ...

__all__ = ["GraniteSpeechFeatureExtractor"]
