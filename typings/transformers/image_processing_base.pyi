from typing import Any, Self, TypeVar

import os

from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .utils import PushToHubMixin

ImageProcessorType = TypeVar("ImageProcessorType", bound=ImageProcessingMixin)
logger = ...

class BatchFeature(BaseBatchFeature): ...

class ImageProcessingMixin(PushToHubMixin):
    _auto_class = ...
    def __init__(self, **kwargs) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = ...,
        force_download: bool = ...,
        local_files_only: bool = ...,
        token: str | bool | None = ...,
        revision: str = ...,
        **kwargs,
    ) -> Self: ...
    def save_pretrained(
        self, save_directory: str | os.PathLike, push_to_hub: bool = ..., **kwargs
    ):  # -> list[str]:
        ...
    @classmethod
    def get_image_processor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...
    @classmethod
    def from_dict(
        cls, image_processor_dict: dict[str, Any], **kwargs
    ):  # -> tuple[Self, dict[str, Any]] | Self:
        ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_json_file(cls, json_file: str | os.PathLike):  # -> Self:
        ...
    def to_json_string(self) -> str: ...
    def to_json_file(self, json_file_path: str | os.PathLike):  # -> None:
        ...
    def __repr__(self):  # -> str:
        ...
    @classmethod
    def register_for_auto_class(cls, auto_class=...):  # -> None:
        ...
    def fetch_images(
        self, image_url_or_urls: str | list[str] | list[list[str]]
    ):  # -> list[list[list[Any] | Image] | Image] | Image:
        ...
