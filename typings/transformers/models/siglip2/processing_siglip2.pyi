from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin

"""
Image/Text processor class for SigLIP2.
"""

class Siglip2ImagesKwargs(ImagesKwargs, total=False):
    max_num_patches: int | None
    patch_size: int | None

class Siglip2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Siglip2ImagesKwargs
    _defaults = ...

class Siglip2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    valid_processor_kwargs = Siglip2ProcessorKwargs
    def __init__(self, image_processor, tokenizer) -> None: ...

__all__ = ["Siglip2Processor"]
