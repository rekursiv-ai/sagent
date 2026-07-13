from ....processing_utils import ProcessorMixin

"""
Processor class for TVLT.
"""

class TvltProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    feature_extractor_class = ...
    def __init__(self, image_processor, feature_extractor) -> None: ...
    def __call__(
        self,
        images=...,
        audio=...,
        images_mixed=...,
        sampling_rate=...,
        mask_audio=...,
        mask_pixel=...,
        *args,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = ["TvltProcessor"]
