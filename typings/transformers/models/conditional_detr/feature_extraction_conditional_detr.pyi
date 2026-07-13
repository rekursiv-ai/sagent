from .image_processing_conditional_detr import ConditionalDetrImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for Conditional DETR."""
logger = ...

def rgb_to_id(x):  # -> NDArray[signedinteger[Any]] | int:
    ...

@requires(backends=("vision",))
class ConditionalDetrFeatureExtractor(ConditionalDetrImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["ConditionalDetrFeatureExtractor"]
