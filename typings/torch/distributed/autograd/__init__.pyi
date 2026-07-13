from types import TracebackType as TracebackType
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Any as Any,
)

import types

from torch._C._distributed_autograd import (
    DistAutogradContext as DistAutogradContext,
    _current_context as _current_context,
    _get_debug_info as _get_debug_info,
    _get_max_id as _get_max_id,
    _init as _init,
    _is_valid_context as _is_valid_context,
    _new_context as _new_context,
    _release_context as _release_context,
    _retrieve_context as _retrieve_context,
    backward as backward,
    get_gradients as get_gradients,
)

def is_available() -> bool: ...

__all__ = ["context", "is_available"]

class context:
    def __enter__(self) -> int: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
