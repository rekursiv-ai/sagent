from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin

"""
Processor class for BridgeTower.
"""

class BridgeTowerImagesKwargs(ImagesKwargs):
    size_divisor: int | None

class BridgeTowerProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: BridgeTowerImagesKwargs
    _defaults = ...

class BridgeTowerProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    valid_processor_kwargs = BridgeTowerProcessorKwargs
    def __init__(self, image_processor, tokenizer) -> None: ...

__all__ = ["BridgeTowerProcessor"]
