from collections.abc import Callable
from typing import Any

from torch.utils._pytree import Context, TreeSpec

def reorder_kwargs(user_kwargs: dict[str, Any], spec: TreeSpec) -> dict[str, Any]: ...
def is_equivalent(
    spec1: TreeSpec,
    spec2: TreeSpec,
    equivalence_fn: Callable[[type | None, Context, type | None, Context], bool],
) -> bool: ...
