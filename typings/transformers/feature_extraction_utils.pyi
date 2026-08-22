from collections import UserDict
from typing import Any, Self, TypeVar

import os

from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .utils import PushToHubMixin, TensorType

"""
Feature extraction saving/loading class for common feature extractors.
"""
logger = ...
type PreTrainedFeatureExtractor = SequenceFeatureExtractor
SpecificFeatureExtractorType = TypeVar(
    "SpecificFeatureExtractorType", bound=FeatureExtractionMixin
)

class BatchFeature(UserDict):
    def __init__(
        self,
        data: dict[str, Any] | None = ...,
        tensor_type: str | TensorType | None = ...,
    ) -> None: ...
    def __getitem__(self, item: str) -> Any: ...
    def __getattr__(self, item: str): ...
    def __getstate__(self):  # -> dict[str, dict[Any, Any]]:
        ...
    def __setstate__(self, state):  # -> None:
        ...
    def convert_to_tensors(
        self, tensor_type: str | TensorType | None = ...
    ):  # -> Self:
        ...
    def to(self, *args, **kwargs) -> BatchFeature: ...

class FeatureExtractionMixin(PushToHubMixin):
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
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...
    @classmethod
    def from_dict(
        cls, feature_extractor_dict: dict[str, Any], **kwargs
    ) -> FeatureExtractionMixin | tuple[FeatureExtractionMixin, dict[str, Any]]: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_json_file(cls, json_file: str | os.PathLike) -> FeatureExtractionMixin: ...
    def to_json_string(self) -> str: ...
    def to_json_file(self, json_file_path: str | os.PathLike):  # -> None:
        ...
    def __repr__(self):  # -> str:
        ...
    @classmethod
    def register_for_auto_class(cls, auto_class=...):  # -> None:
        ...
