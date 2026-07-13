from contextlib import contextmanager

from ...processing_utils import ProcessorMixin

"""
Speech processor class for Speech2Text
"""

class Speech2TextProcessor(ProcessorMixin):
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer) -> None: ...
    def __call__(self, *args, **kwargs): ...
    @contextmanager
    def as_target_processor(self):  # -> Generator[None, Any, None]:
        ...

__all__ = ["Speech2TextProcessor"]
