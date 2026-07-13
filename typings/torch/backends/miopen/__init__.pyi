from contextlib import contextmanager as contextmanager

from torch.backends import (
    ContextProp as ContextProp,
    PropModule as PropModule,
    __allow_nonbracketed_mutation as __allow_nonbracketed_mutation,
)

def set_flags(_immediate=...) -> tuple[bool]: ...
@contextmanager
def flags(immediate=...) -> Generator[None, Any, None]: ...

class MiopenModule(PropModule):
    def __init__(self, m, name) -> None: ...

    immediate = ...

immediate: bool
