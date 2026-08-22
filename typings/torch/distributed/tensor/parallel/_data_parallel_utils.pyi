from typing import no_type_check

from torch import Tensor

@no_type_check
def sync_grad_hook(grad, *, device_handle=..., compute_stream=...) -> Tensor: ...
