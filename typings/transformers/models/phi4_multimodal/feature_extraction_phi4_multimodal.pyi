from ...audio_utils import AudioInput
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...image_processing_utils import BatchFeature
from ...utils import TensorType

"""
Processor class for Phi4Multimodal
"""
logger = ...

class Phi4MultimodalFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ...
    def __init__(
        self,
        feature_size: int = ...,
        sampling_rate: int = ...,
        hop_length: int = ...,
        n_fft: int = ...,
        win_length: int = ...,
        preemphasis: float = ...,
        padding_value: float = ...,
        audio_compression_rate: int = ...,
        audio_downsample_rate: int = ...,
        audio_feat_stride: int = ...,
        mel_min_frequency: float = ...,
        mel_max_frequency: float = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        raw_speech: AudioInput,
        sampling_rate: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        padding: str | None = ...,
        max_length: int | None = ...,
        truncation: bool = ...,
        return_tensors: str | TensorType | None = ...,
        return_attention_mask: bool | None = ...,
        device: str | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["Phi4MultimodalFeatureExtractor"]
