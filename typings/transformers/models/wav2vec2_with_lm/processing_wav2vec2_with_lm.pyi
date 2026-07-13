from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pool

from pyctcdecode import BeamSearchDecoderCTC

import numpy as np

from ...feature_extraction_utils import FeatureExtractionMixin
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import PreTrainedTokenizerBase
from ...utils import ModelOutput

"""
Speech processor class for Wav2Vec2
"""
logger = ...
type ListOfDict = list[dict[str, int | str]]

@dataclass
class Wav2Vec2DecoderWithLMOutput(ModelOutput):
    text: list[list[str]] | list[str] | str
    logit_score: list[list[float]] | list[float] | float = ...
    lm_score: list[list[float]] | list[float] | float = ...
    word_offsets: list[list[ListOfDict]] | list[ListOfDict] | ListOfDict = ...

class Wav2Vec2ProcessorWithLM(ProcessorMixin):
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        feature_extractor: FeatureExtractionMixin,
        tokenizer: PreTrainedTokenizerBase,
        decoder: BeamSearchDecoderCTC,
    ) -> None: ...
    def save_pretrained(self, save_directory):  # -> None:
        ...
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):  # -> Self:
        ...
    @property
    def language_model(self): ...
    @staticmethod
    def get_missing_alphabet_tokens(decoder, tokenizer):  # -> set[Any]:
        ...
    def __call__(self, *args, **kwargs): ...
    def pad(self, *args, **kwargs): ...
    def batch_decode(
        self,
        logits: np.ndarray,
        pool: Pool | None = ...,
        num_processes: int | None = ...,
        beam_width: int | None = ...,
        beam_prune_logp: float | None = ...,
        token_min_logp: float | None = ...,
        hotwords: Iterable[str] | None = ...,
        hotword_weight: float | None = ...,
        alpha: float | None = ...,
        beta: float | None = ...,
        unk_score_offset: float | None = ...,
        lm_score_boundary: bool | None = ...,
        output_word_offsets: bool = ...,
        n_best: int = ...,
    ):  # -> Wav2Vec2DecoderWithLMOutput:
        ...
    def decode(
        self,
        logits: np.ndarray,
        beam_width: int | None = ...,
        beam_prune_logp: float | None = ...,
        token_min_logp: float | None = ...,
        hotwords: Iterable[str] | None = ...,
        hotword_weight: float | None = ...,
        alpha: float | None = ...,
        beta: float | None = ...,
        unk_score_offset: float | None = ...,
        lm_score_boundary: bool | None = ...,
        output_word_offsets: bool = ...,
        n_best: int = ...,
    ):  # -> Wav2Vec2DecoderWithLMOutput:
        ...
    @contextmanager
    def as_target_processor(self):  # -> Generator[None, Any, None]:
        ...

__all__ = ["Wav2Vec2ProcessorWithLM"]
