from .image_processing_yolos import YolosImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for YOLOS."""
logger = ...

def rgb_to_id(x):  # -> NDArray[signedinteger[Any]] | int:
    ...

@requires(backends=("vision",))
class YolosFeatureExtractor(YolosImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["YolosFeatureExtractor"]
