from torch import Tensor
from torch.types import _bool

class ViewMeta:
    has_symbolic_inputs: _bool

def get_view_meta_sequence(tensor: Tensor) -> list[ViewMeta]: ...
def apply_view_meta_sequence(base: Tensor, sequence: list[ViewMeta]) -> Tensor: ...
