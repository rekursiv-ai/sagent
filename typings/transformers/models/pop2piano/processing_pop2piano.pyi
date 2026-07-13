import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import BatchEncoding, PaddingStrategy, TruncationStrategy
from ...utils import TensorType
from ...utils.import_utils import requires

"""Processor class for Pop2Piano."""

@requires(backends=("essentia", "librosa", "pretty_midi", "scipy", "torch"))
class Pop2PianoProcessor(ProcessorMixin):
    attributes = ...
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer) -> None: ...
    def __call__(
        self,
        audio: np.ndarray | list[float] | list[np.ndarray] = ...,
        sampling_rate: int | list[int] | None = ...,
        steps_per_beat: int = ...,
        resample: bool | None = ...,
        notes: list | TensorType = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchFeature | BatchEncoding: ...
    def batch_decode(
        self, token_ids, feature_extractor_output: BatchFeature, return_midi: bool = ...
    ) -> BatchEncoding: ...
    def save_pretrained(self, save_directory, **kwargs):  # -> list[str]:
        ...
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):  # -> Self:
        ...

__all__ = ["Pop2PianoProcessor"]
