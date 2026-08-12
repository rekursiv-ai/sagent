from torch import Tensor
from typing import no_type_check

@no_type_check
def sync_grad_hook(grad, *, device_handle=..., compute_stream=...) -> Tensor: ...
