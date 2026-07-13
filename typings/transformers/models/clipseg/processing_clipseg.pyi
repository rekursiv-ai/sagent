from ...processing_utils import ProcessorMixin

"""
Image/Text processor class for CLIPSeg
"""

class CLIPSegProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    def __call__(
        self, text=..., images=..., visual_prompt=..., return_tensors=..., **kwargs
    ):  # -> dict[str, Any] | BatchEncoding:
        ...
    @property
    def feature_extractor_class(
        self,
    ):  # -> tuple[Literal['ViTImageProcessor'], Literal['ViTImageProcessorFast']]:
        ...
    @property
    def feature_extractor(self): ...

__all__ = ["CLIPSegProcessor"]
