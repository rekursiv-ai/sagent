from ...processing_utils import ProcessorMixin

"""
Image/Text processor class for CLIP
"""

class CLIPProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    @property
    def feature_extractor_class(
        self,
    ):  # -> tuple[Literal['CLIPImageProcessor'], Literal['CLIPImageProcessorFast']]:
        ...
    @property
    def feature_extractor(self): ...

__all__ = ["CLIPProcessor"]
