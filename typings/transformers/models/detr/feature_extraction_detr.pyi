from .image_processing_detr import DetrImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for DETR."""
logger = ...

def rgb_to_id(x):  # -> NDArray[signedinteger[Any]] | int:
    ...

@requires(backends=("vision",))
class DetrFeatureExtractor(DetrImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["DetrFeatureExtractor"]
