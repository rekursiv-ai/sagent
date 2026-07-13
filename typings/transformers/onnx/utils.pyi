from ctypes import c_float
from enum import Enum

from .. import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

class ParameterFormat(Enum):
    Float = c_float
    @property
    def size(self) -> int: ...

def compute_effective_axis_dimension(
    dimension: int, fixed_dimension: int, num_token_to_add: int = ...
) -> int: ...
def compute_serialized_parameters_size(
    num_parameters: int, dtype: ParameterFormat
) -> int: ...
def get_preprocessor(
    model_name: str,
) -> AutoTokenizer | AutoFeatureExtractor | AutoProcessor | None: ...
