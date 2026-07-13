from collections.abc import Generator

import contextlib

from torch.utils._content_store import ContentStoreReader

LOAD_TENSOR_READER: ContentStoreReader | None = ...

@contextlib.contextmanager
def load_tensor_reader(loc: str) -> Generator[None]: ...
def register_debug_prims() -> None: ...
