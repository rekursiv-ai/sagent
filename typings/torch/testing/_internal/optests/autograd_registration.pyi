from collections.abc import Generator
from typing import Any

import contextlib

@contextlib.contextmanager
def set_autograd_fallback_mode(mode) -> Generator[None, Any]: ...
def autograd_registration_check(op, args, kwargs) -> None: ...
