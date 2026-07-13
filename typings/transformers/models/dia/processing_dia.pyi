from pathlib import Path

import torch

from ...audio_utils import AudioInput
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack

"""Processor class for Dia"""

class DiaAudioKwargs(AudioKwargs, total=False):
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    delay_pattern: list[int]
    generation: bool

class DiaProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: DiaAudioKwargs
    _defaults = ...

class DiaProcessor(ProcessorMixin):
    feature_extractor_class = ...
    tokenizer_class = ...
    audio_tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer, audio_tokenizer) -> None: ...
    def __call__(
        self,
        text: str | list[str],
        audio: AudioInput | None = ...,
        output_labels: bool | None = ...,
        **kwargs: Unpack[DiaProcessorKwargs],
    ):  # -> BatchFeature:
        ...
    def batch_decode(
        self,
        decoder_input_ids: torch.Tensor,
        audio_prompt_len: int | None = ...,
        **kwargs: Unpack[DiaProcessorKwargs],
    ) -> list[torch.Tensor]: ...
    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        audio_prompt_len: int | None = ...,
        **kwargs: Unpack[DiaProcessorKwargs],
    ) -> torch.Tensor: ...
    def get_audio_prompt_len(
        self, decoder_attention_mask: torch.Tensor, **kwargs: Unpack[DiaProcessorKwargs]
    ) -> int: ...
    def save_audio(
        self,
        audio: AudioInput,
        saving_path: str | Path | list[str | Path],
        **kwargs: Unpack[DiaProcessorKwargs],
    ):  # -> None:
        ...
    @staticmethod
    def build_indices(
        bsz: int,
        seq_len: int,
        num_channels: int,
        delay_pattern: list[int],
        revert: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def apply_audio_delay(
        audio: torch.Tensor,
        pad_token_id: int,
        bos_token_id: int,
        precomputed_idx: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor: ...

__all__ = ["DiaProcessor"]
