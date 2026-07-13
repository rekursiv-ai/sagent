from typing import Self

import types

from filelock import FileLock as base_FileLock

class FileLock(base_FileLock):
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
