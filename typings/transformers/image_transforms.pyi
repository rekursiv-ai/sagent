from collections.abc import Collection, Iterable

import jax.numpy as jnp
import numpy as np
import PIL
import tensorflow as tf
import torch

from .image_utils import ChannelDimension, ImageInput, PILImageResampling
from .utils import ExplicitEnum, TensorType

def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: ChannelDimension | str,
    input_channel_dim: ChannelDimension | str | None = ...,
) -> np.ndarray: ...
def rescale(
    image: np.ndarray,
    scale: float,
    data_format: ChannelDimension | None = ...,
    dtype: np.dtype = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def to_pil_image(
    image: np.ndarray | PIL.Image.Image | torch.Tensor | tf.Tensor | jnp.ndarray,
    do_rescale: bool | None = ...,
    image_mode: str | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> PIL.Image.Image: ...
def get_size_with_aspect_ratio(image_size, size, max_size=...) -> tuple[int, int]: ...
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: int | tuple[int, int] | list[int] | tuple[int, ...],
    default_to_square: bool = ...,
    max_size: int | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple: ...
def resize(
    image: np.ndarray,
    size: tuple[int, int],
    resample: PILImageResampling | None = ...,
    reducing_gap: int | None = ...,
    data_format: ChannelDimension | None = ...,
    return_numpy: bool = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def normalize(
    image: np.ndarray,
    mean: float | Collection[float],
    std: float | Collection[float],
    data_format: ChannelDimension | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def center_crop(
    image: np.ndarray,
    size: tuple[int, int],
    data_format: str | ChannelDimension | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def center_to_corners_format(bboxes_center: TensorType) -> TensorType: ...
def corners_to_center_format(bboxes_corners: TensorType) -> TensorType: ...
def rgb_to_id(color):  # -> NDArray[signedinteger[Any]] | int:
    ...
def id_to_rgb(id_map):  # -> NDArray[unsignedinteger[_8Bit]] | list[Any]:
    ...

class PaddingMode(ExplicitEnum):
    CONSTANT = ...
    REFLECT = ...
    REPLICATE = ...
    SYMMETRIC = ...

def pad(
    image: np.ndarray,
    padding: int | tuple[int, int] | Iterable[tuple[int, int]],
    mode: PaddingMode = ...,
    constant_values: float | Iterable[float] = ...,
    data_format: str | ChannelDimension | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def convert_to_rgb(image: ImageInput) -> ImageInput: ...
def flip_channel_order(
    image: np.ndarray,
    data_format: ChannelDimension | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def group_images_by_shape(
    images: list[torch.Tensor] | torch.Tensor,
    disable_grouping: bool,
    is_nested: bool = ...,
) -> tuple[
    dict[tuple[int, int], list[torch.Tensor]],
    dict[int | tuple[int, int], tuple[tuple[int, int], int]],
]: ...
def reorder_images(
    processed_images: dict[tuple[int, int], torch.Tensor],
    grouped_images_index: dict[int | tuple[int, int], tuple[tuple[int, int], int]],
    is_nested: bool = ...,
) -> list[torch.Tensor] | torch.Tensor: ...

class NumpyToTensor:
    def __call__(self, image: np.ndarray):  # -> Tensor:
        ...
