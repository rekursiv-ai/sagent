from collections.abc import Sequence
from typing import Any

import io

import numpy as np
import torch

from .utils import is_torchcodec_available

"""
Audio processing functions to extract features from audio waveforms. This code is pure numpy to support all frameworks
and remove unnecessary dependencies.
"""
if is_torchcodec_available():
    TORCHCODEC_VERSION = ...
type AudioInput = (
    np.ndarray | torch.Tensor | Sequence[np.ndarray] | Sequence[torch.Tensor]
)

def load_audio(
    audio: str | np.ndarray, sampling_rate=..., timeout=...
) -> np.ndarray: ...
def load_audio_torchcodec(audio: str | np.ndarray, sampling_rate=...) -> np.ndarray: ...
def load_audio_librosa(
    audio: str | np.ndarray, sampling_rate=..., timeout=...
) -> np.ndarray: ...
def load_audio_as(
    audio: str,
    return_format: str,
    timeout: int | None = ...,
    force_mono: bool = ...,
    sampling_rate: int | None = ...,
) -> str | dict[str, Any] | io.BytesIO | None: ...
def is_valid_audio(audio):  # -> bool:
    ...
def is_valid_list_of_audio(audio):  # -> bool:
    ...
def make_list_of_audio(audio: list[AudioInput] | AudioInput) -> AudioInput: ...
def hertz_to_mel(
    freq: float | np.ndarray, mel_scale: str = ...
) -> float | np.ndarray: ...
def mel_to_hertz(
    mels: float | np.ndarray, mel_scale: str = ...
) -> float | np.ndarray: ...
def hertz_to_octave(
    freq: float | np.ndarray, tuning: float = ..., bins_per_octave: int = ...
):  # -> NDArray[Any]:
    ...
def chroma_filter_bank(
    num_frequency_bins: int,
    num_chroma: int,
    sampling_rate: int,
    tuning: float = ...,
    power: float | None = ...,
    weighting_parameters: tuple[float, float] | None = ...,
    start_at_c_chroma: bool = ...,
):  # -> NDArray[Any]:
    ...
def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: str | None = ...,
    mel_scale: str = ...,
    triangularize_in_mel_space: bool = ...,
) -> np.ndarray: ...
def optimal_fft_length(window_length: int) -> int: ...
def window_function(
    window_length: int,
    name: str = ...,
    periodic: bool = ...,
    frame_length: int | None = ...,
    center: bool = ...,
) -> np.ndarray: ...
def spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: int | None = ...,
    power: float | None = ...,
    center: bool = ...,
    pad_mode: str = ...,
    onesided: bool = ...,
    dither: float = ...,
    preemphasis: float | None = ...,
    mel_filters: np.ndarray | None = ...,
    mel_floor: float = ...,
    log_mel: str | None = ...,
    reference: float = ...,
    min_value: float = ...,
    db_range: float | None = ...,
    remove_dc_offset: bool = ...,
    dtype: np.dtype = ...,
) -> np.ndarray: ...
def spectrogram_batch(
    waveform_list: list[np.ndarray],
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: int | None = ...,
    power: float | None = ...,
    center: bool = ...,
    pad_mode: str = ...,
    onesided: bool = ...,
    dither: float = ...,
    preemphasis: float | None = ...,
    mel_filters: np.ndarray | None = ...,
    mel_floor: float = ...,
    log_mel: str | None = ...,
    reference: float = ...,
    min_value: float = ...,
    db_range: float | None = ...,
    remove_dc_offset: bool = ...,
    dtype: np.dtype = ...,
) -> list[np.ndarray]: ...
def power_to_db(
    spectrogram: np.ndarray,
    reference: float = ...,
    min_value: float = ...,
    db_range: float | None = ...,
) -> np.ndarray: ...
def power_to_db_batch(
    spectrogram: np.ndarray,
    reference: float = ...,
    min_value: float = ...,
    db_range: float | None = ...,
) -> np.ndarray: ...
def amplitude_to_db(
    spectrogram: np.ndarray,
    reference: float = ...,
    min_value: float = ...,
    db_range: float | None = ...,
) -> np.ndarray: ...
def amplitude_to_db_batch(
    spectrogram: np.ndarray,
    reference: float = ...,
    min_value: float = ...,
    db_range: float | None = ...,
) -> np.ndarray: ...
