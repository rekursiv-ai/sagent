from ...processing_utils import ProcessingKwargs, ProcessorMixin

"""
Processor class for Bros.
"""

class BrosProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class BrosProcessor(ProcessorMixin):
    attributes = ...
    tokenizer_class = ...
    valid_processor_kwargs = BrosProcessorKwargs
    def __init__(self, tokenizer=..., **kwargs) -> None: ...

__all__ = ["BrosProcessor"]
