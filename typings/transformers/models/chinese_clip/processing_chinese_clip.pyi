from ...processing_utils import ProcessorMixin

"""
Image/Text processor class for Chinese-CLIP
"""

class ChineseCLIPProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    @property
    def feature_extractor_class(
        self,
    ):  # -> tuple[Literal['ChineseCLIPImageProcessor'], Literal['ChineseCLIPImageProcessorFast']]:
        ...

__all__ = ["ChineseCLIPProcessor"]
