import numpy as np
import pretty_midi

from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import (
    BatchEncoding,
    PaddingStrategy,
    PreTrainedTokenizer,
    TruncationStrategy,
)
from ...utils import TensorType
from ...utils.import_utils import requires

"""Tokenization class for Pop2Piano."""
logger = ...
VOCAB_FILES_NAMES = ...

def token_time_to_note(number, cutoff_time_idx, current_idx): ...
def token_note_to_note(
    number, current_velocity, default_velocity, note_onsets_ready, current_idx, notes
): ...

@requires(backends=("pretty_midi", "torch"))
class Pop2PianoTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab,
        default_velocity=...,
        num_bars=...,
        unk_token=...,
        eos_token=...,
        pad_token=...,
        bos_token=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, Any]:
        ...
    def relative_batch_tokens_ids_to_notes(
        self,
        tokens: np.ndarray,
        beat_offset_idx: int,
        bars_per_batch: int,
        cutoff_time_idx: int,
    ):  # -> list[Any] | ndarray[Any, dtype[Any]]:
        ...
    def relative_batch_tokens_ids_to_midi(
        self,
        tokens: np.ndarray,
        beatstep: np.ndarray,
        beat_offset_idx: int = ...,
        bars_per_batch: int = ...,
        cutoff_time_idx: int = ...,
    ): ...
    def relative_tokens_ids_to_notes(
        self,
        tokens: np.ndarray,
        start_idx: float,
        cutoff_time_idx: float | None = ...,
    ):  # -> list[Any] | ndarray[Any, dtype[Any]]:
        ...
    def notes_to_midi(
        self, notes: np.ndarray, beatstep: np.ndarray, offset_sec: int = ...
    ): ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def encode_plus(
        self,
        notes: np.ndarray | list[pretty_midi.Note],
        truncation_strategy: TruncationStrategy | None = ...,
        max_length: int | None = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def batch_encode_plus(
        self,
        notes: np.ndarray | list[pretty_midi.Note],
        truncation_strategy: TruncationStrategy | None = ...,
        max_length: int | None = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def __call__(
        self,
        notes: np.ndarray | list[pretty_midi.Note] | list[list[pretty_midi.Note]],
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        return_attention_mask: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def batch_decode(
        self, token_ids, feature_extractor_output: BatchFeature, return_midi: bool = ...
    ):  # -> BatchEncoding:
        ...

__all__ = ["Pop2PianoTokenizer"]
