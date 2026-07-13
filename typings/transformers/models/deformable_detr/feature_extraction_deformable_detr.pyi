from .image_processing_deformable_detr import DeformableDetrImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for Deformable DETR."""
logger = ...

def rgb_to_id(x):  # -> NDArray[signedinteger[Any]] | int:
    ...

@requires(backends=("vision",))
class DeformableDetrFeatureExtractor(DeformableDetrImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["DeformableDetrFeatureExtractor"]
