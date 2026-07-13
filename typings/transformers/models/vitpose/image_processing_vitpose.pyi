import numpy as np
import PIL

from .modeling_vitpose import VitPoseEstimatorOutput
from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput
from ...utils import (
    TensorType,
)

"""Image processor class for VitPose."""
logger = ...

def box_to_center_and_scale(
    box: tuple | list | np.ndarray,
    image_width: int,
    image_height: int,
    normalize_factor: float = ...,
    padding_factor: float = ...,
):  # -> tuple[NDArray[floating[_32Bit]], NDArray[floating[Any]]]:
    ...
def coco_to_pascal_voc(bboxes: np.ndarray) -> np.ndarray: ...
def get_keypoint_predictions(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
def post_dark_unbiased_data_processing(
    coords: np.ndarray, batch_heatmaps: np.ndarray, kernel: int = ...
) -> np.ndarray: ...
def transform_preds(
    coords: np.ndarray, center: np.ndarray, scale: np.ndarray, output_size: np.ndarray
) -> np.ndarray: ...
def get_warp_matrix(
    theta: float, size_input: np.ndarray, size_dst: np.ndarray, size_target: np.ndarray
):  # -> NDArray[floating[_32Bit]]:
    ...
def scipy_warp_affine(src, M, size):  # -> NDArray[Any]:
    ...

class VitPoseImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_affine_transform: bool = ...,
        size: dict[str, int] | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        **kwargs,
    ) -> None: ...
    def affine_transform(
        self,
        image: np.ndarray,
        center: tuple[float],
        scale: tuple[float],
        rotation: float,
        size: dict[str, int],
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def preprocess(
        self,
        images: ImageInput,
        boxes: list[list[float]] | np.ndarray,
        do_affine_transform: bool | None = ...,
        size: dict[str, int] | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...
    def keypoints_from_heatmaps(
        self,
        heatmaps: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        kernel: int = ...,
    ):  # -> tuple[ndarray[Any, Any], ndarray[Any, Any]]:
        ...
    def post_process_pose_estimation(
        self,
        outputs: VitPoseEstimatorOutput,
        boxes: list[list[list[float]]] | np.ndarray,
        kernel_size: int = ...,
        threshold: float | None = ...,
        target_sizes: TensorType | list[tuple] = ...,
    ):  # -> list[list[dict[str, Tensor]]]:
        ...

__all__ = ["VitPoseImageProcessor"]
