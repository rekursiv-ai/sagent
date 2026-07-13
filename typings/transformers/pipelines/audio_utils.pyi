import numpy as np

def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.ndarray: ...
def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = ...,
    ffmpeg_input_device: str | None = ...,
    ffmpeg_additional_args: list[str] | None = ...,
):  # -> Generator[bytes, Any, None]:
    ...
def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: int | None = ...,
    stride_length_s: tuple[float, float] | float | None = ...,
    format_for_conversion: str = ...,
    ffmpeg_input_device: str | None = ...,
    ffmpeg_additional_args: list[str] | None = ...,
):  # -> Generator[dict[str, bytes | tuple[int, Literal[0]] | bool] | dict[str, bytes | tuple[int, int]], Any, None]:
    ...
def chunk_bytes_iter(
    iterator, chunk_len: int, stride: tuple[int, int], stream: bool = ...
):  # -> Generator[dict[str, Any | tuple[int, Literal[0]] | bool] | dict[str, Any | tuple[int, int]] | dict[str, Any | bytes | tuple[int, int]], Any, None]:
    ...
