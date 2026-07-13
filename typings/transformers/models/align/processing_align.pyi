from ...processing_utils import ProcessingKwargs, ProcessorMixin

"""
Image/Text processor class for ALIGN
"""

class AlignProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class AlignProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    valid_processor_kwargs = AlignProcessorKwargs
    def __init__(self, image_processor, tokenizer) -> None: ...

__all__ = ["AlignProcessor"]
