from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .utils import PaddingStrategy, TensorType

"""
Sequence feature extraction class for common feature extractors to preprocess sequences.
"""
logger = ...

class SequenceFeatureExtractor(FeatureExtractionMixin):
    def __init__(
        self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs
    ) -> None: ...
    def pad(
        self,
        processed_features: BatchFeature
        | list[BatchFeature]
        | dict[str, BatchFeature]
        | dict[str, list[BatchFeature]]
        | list[dict[str, BatchFeature]],
        padding: bool | str | PaddingStrategy = ...,
        max_length: int | None = ...,
        truncation: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_attention_mask: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
    ) -> BatchFeature: ...
