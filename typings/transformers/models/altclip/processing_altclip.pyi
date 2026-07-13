from ...processing_utils import ProcessorMixin
from ...utils.deprecation import deprecate_kwarg

"""
Image/Text processor class for AltCLIP
"""

class AltCLIPProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    @deprecate_kwarg(
        old_name="feature_extractor", version="5.0.0", new_name="image_processor"
    )
    def __init__(self, image_processor=..., tokenizer=...) -> None: ...

__all__ = ["AltCLIPProcessor"]
