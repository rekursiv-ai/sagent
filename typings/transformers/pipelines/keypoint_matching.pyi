from collections.abc import Sequence
from typing import Any, TypedDict, overload

from PIL import Image

from .base import Pipeline

type ImagePair = Sequence[Image.Image | str]

class Keypoint(TypedDict):
    x: float
    y: float

class Match(TypedDict):
    keypoint_image_0: Keypoint
    keypoint_image_1: Keypoint
    score: float

def validate_image_pairs(images: Any) -> Sequence[Sequence[ImagePair]]: ...

class KeypointMatchingPipeline(Pipeline):
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    def __call__(
        self, inputs: ImagePair, threshold: float = ..., **kwargs: Any
    ) -> list[Match]: ...
    @overload
    def __call__(
        self, inputs: list[ImagePair], threshold: float = ..., **kwargs: Any
    ) -> list[list[Match]]: ...
    def __call__(
        self,
        inputs: list[ImagePair] | ImagePair,
        threshold: float = ...,
        **kwargs: Any,
    ) -> list[Match] | list[list[Match]]: ...
    def preprocess(
        self, images, timeout=...
    ):  # -> dict[str, BatchFeature | Any | list[tuple[int, int]]]:
        ...
    def postprocess(self, forward_outputs, threshold=...) -> list[Match]: ...
