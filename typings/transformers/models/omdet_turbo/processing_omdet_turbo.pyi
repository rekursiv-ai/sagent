from .modeling_omdet_turbo import OmDetTurboObjectDetectionOutput
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType
from ...utils.import_utils import requires

"""
Processor class for OmDet-Turbo.
"""

class OmDetTurboTextKwargs(TextKwargs, total=False):
    task: str | list[str] | TextInput | PreTokenizedInput | None

class OmDetTurboProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: OmDetTurboTextKwargs
    _defaults = ...

class DictWithDeprecationWarning(dict):
    message = ...
    def __getitem__(self, key): ...
    def get(self, key, *args, **kwargs): ...

def clip_boxes(box, box_size: tuple[int, int]):  # -> Tensor:
    ...
def compute_score(boxes):  # -> tuple[Tensor, Tensor]:
    ...

@requires(backends=("vision", "torchvision"))
class OmDetTurboProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: list[str] | list[list[str]] | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[OmDetTurboProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def model_input_names(self): ...
    def post_process_grounded_object_detection(
        self,
        outputs: OmDetTurboObjectDetectionOutput,
        text_labels: list[str] | list[list[str]] | None = ...,
        threshold: float = ...,
        nms_threshold: float = ...,
        target_sizes: TensorType | list[tuple] | None = ...,
        max_num_det: int | None = ...,
    ):  # -> list[Any]:
        ...

__all__ = ["OmDetTurboProcessor"]
