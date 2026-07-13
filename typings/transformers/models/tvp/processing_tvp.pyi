from ...processing_utils import ProcessingKwargs, ProcessorMixin

"""
Processor class for TVP.
"""

class TvpProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class TvpProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    def post_process_video_grounding(
        self, logits, video_durations
    ):  # -> tuple[Any, Any]:
        ...

__all__ = ["TvpProcessor"]
