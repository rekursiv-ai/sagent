import pathlib

from .modeling_grounding_dino import GroundingDinoObjectDetectionOutput
from ...image_utils import AnnotationFormat, ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import TensorType

"""
Processor class for Grounding DINO.
"""
type AnnotationType = dict[str, int | str | list[dict]]

def get_phrases_from_posmap(posmaps, input_ids):  # -> list[Any]:
    ...

class DictWithDeprecationWarning(dict):
    message = ...
    def __getitem__(self, key): ...
    def get(self, key, *args, **kwargs): ...

class GroundingDinoImagesKwargs(ImagesKwargs, total=False):
    annotations: AnnotationType | list[AnnotationType] | None
    return_segmentation_masks: bool | None
    masks_path: str | pathlib.Path | None
    do_convert_annotations: bool | None
    format: str | AnnotationFormat | None

class GroundingDinoProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: GroundingDinoImagesKwargs
    _defaults = ...

class GroundingDinoProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    valid_processor_kwargs = GroundingDinoProcessorKwargs
    def __init__(self, image_processor, tokenizer) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        **kwargs: Unpack[GroundingDinoProcessorKwargs],
    ) -> BatchEncoding: ...
    def post_process_grounded_object_detection(
        self,
        outputs: GroundingDinoObjectDetectionOutput,
        input_ids: TensorType | None = ...,
        threshold: float = ...,
        text_threshold: float = ...,
        target_sizes: TensorType | list[tuple] | None = ...,
        text_labels: list[list[str]] | None = ...,
    ):  # -> list[Any]:
        ...

__all__ = ["GroundingDinoProcessor"]
