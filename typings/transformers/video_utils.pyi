from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import NewType

import numpy as np
import PIL.Image
import torch

from .image_transforms import PaddingMode
from .image_utils import ChannelDimension

logger = ...
URL = NewType("URL", str)
Path = NewType("Path", str)
type VideoInput = (
    list[PIL.Image.Image]
    | np.ndarray
    | torch.Tensor
    | list[np.ndarray]
    | list[torch.Tensor]
    | list[list[PIL.Image.Image]]
    | list[list[np.ndarray]]
    | list[list[torch.Tensor]]
    | URL
    | list[URL]
    | list[list[URL]]
    | Path
    | list[Path]
    | list[list[Path]]
)

@dataclass
class VideoMetadata(Mapping):
    total_num_frames: int
    fps: float | None = ...
    width: int | None = ...
    height: int | None = ...
    duration: float | None = ...
    video_backend: str | None = ...
    frames_indices: list[int] | None = ...
    def __iter__(self):  # -> Generator[str, None, None]:
        ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, item):  # -> Any:
        ...
    def __setitem__(self, key, value):  # -> None:
        ...
    @property
    def timestamps(self) -> list[float]: ...
    def update(self, dictionary):  # -> None:
        ...

def is_valid_video_frame(frame):  # -> bool:
    ...
def is_valid_video(video):  # -> list[Any] | bool | tuple[Any, ...]:
    ...
def valid_videos(videos):  # -> bool:
    ...
def is_batched_video(videos):  # -> list[Any] | bool | tuple[Any, ...]:
    ...
def is_scaled_video(video: np.ndarray) -> bool: ...
def convert_pil_frames_to_video(
    videos: list[VideoInput],
) -> list[np.ndarray | torch.Tensor]: ...
def make_batched_videos(videos) -> list[np.ndarray | torch.Tensor | URL | Path]: ...
def make_batched_metadata(
    videos: VideoInput, video_metadata: VideoMetadata | dict
):  # -> VideoMetadata | dict[Any, Any] | <subclass of VideoMetadata and list> | <subclass of dict and list>:
    ...
def get_video_size(
    video: np.ndarray, channel_dim: ChannelDimension | None = ...
) -> tuple[int, int]: ...
def get_uniform_frame_indices(
    total_num_frames: int, num_frames: int | None = ...
):  # -> NDArray[Any]:
    ...
def default_sample_indices_fn(
    metadata: VideoMetadata, num_frames=..., fps=..., **kwargs
):  # -> NDArray[Any]:
    ...
def read_video_opencv(
    video_path: URL | Path, sample_indices_fn: Callable, **kwargs
) -> tuple[np.ndarray, VideoMetadata]: ...
def read_video_decord(
    video_path: URL | Path, sample_indices_fn: Callable, **kwargs
):  # -> tuple[Any, VideoMetadata]:
    ...
def read_video_pyav(
    video_path: URL | Path, sample_indices_fn: Callable, **kwargs
):  # -> tuple[NDArray[Any], VideoMetadata]:
    ...
def read_video_torchvision(
    video_path: URL | Path, sample_indices_fn: Callable, **kwargs
):  # -> tuple[Any, VideoMetadata]:
    ...
def read_video_torchcodec(
    video_path: URL | Path, sample_indices_fn: Callable, **kwargs
):  # -> tuple[Tensor, VideoMetadata]:
    ...

VIDEO_DECODERS = ...

def load_video(
    video: VideoInput,
    num_frames: int | None = ...,
    fps: float | None = ...,
    backend: str = ...,
    sample_indices_fn: Callable | None = ...,
    **kwargs,
) -> np.ndarray: ...
def convert_to_rgb(
    video: np.ndarray, input_data_format: str | ChannelDimension | None = ...
) -> np.ndarray: ...
def pad(
    video: np.ndarray,
    padding: int | tuple[int, int] | Iterable[tuple[int, int]],
    mode: PaddingMode = ...,
    constant_values: float | Iterable[float] = ...,
    data_format: str | ChannelDimension | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def group_videos_by_shape(
    videos: list[torch.Tensor],
) -> tuple[
    dict[tuple[int, int], torch.Tensor], dict[int, tuple[tuple[int, int], int]]
]: ...
def reorder_videos(
    processed_videos: dict[tuple[int, int], torch.Tensor],
    grouped_videos_index: dict[int, tuple[tuple[int, int], int]],
) -> list[torch.Tensor]: ...
