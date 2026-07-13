import numpy as np
import torch

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs

"""Image processor class for Fuyu."""
logger = ...

def make_list_of_list_of_images(
    images: list[list[ImageInput]] | list[ImageInput] | ImageInput,
) -> list[list[ImageInput]]: ...

class FuyuBatchFeature(BatchFeature):
    def convert_to_tensors(
        self, tensor_type: str | TensorType | None = ...
    ):  # -> Self:
        ...
    def to(self, *args, **kwargs) -> BatchFeature: ...

class FuyuImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_pad: bool = ...,
        padding_value: float = ...,
        padding_mode: str = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] = ...,
        image_std: float | list[float] = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        patch_size: dict[str, int] | None = ...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def pad_image(
        self,
        image: np.ndarray,
        size: dict[str, int],
        mode: str = ...,
        constant_values: float = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_pad: bool | None = ...,
        padding_value: float | None = ...,
        padding_mode: str | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | None = ...,
        image_std: float | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        patch_size: dict[str, int] | None = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        return_tensors: TensorType | None = ...,
    ):  # -> FuyuBatchFeature:
        ...
    def get_num_patches(
        self,
        image_height: int,
        image_width: int,
        patch_size: dict[str, int] | None = ...,
    ) -> int: ...
    def patchify_image(
        self, image: torch.Tensor, patch_size: dict[str, int] | None = ...
    ) -> torch.Tensor: ...
    def preprocess_with_tokenizer_info(
        self,
        image_input: torch.Tensor,
        image_present: torch.Tensor,
        image_unpadded_h: torch.Tensor,
        image_unpadded_w: torch.Tensor,
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
        patch_size: dict[str, int] | None = ...,
    ) -> FuyuBatchFeature: ...

__all__ = ["FuyuImageProcessor"]
