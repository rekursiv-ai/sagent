from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin

"""
Processor class for ViLT.
"""

class ViltImagesKwargs(ImagesKwargs):
    size_divisor: int | None

class ViltProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: ViltImagesKwargs
    _defaults = ...

class ViltProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    valid_processor_kwargs = ViltProcessorKwargs
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    @property
    def feature_extractor_class(self):  # -> str:
        ...
    @property
    def feature_extractor(self): ...

__all__ = ["ViltProcessor"]
