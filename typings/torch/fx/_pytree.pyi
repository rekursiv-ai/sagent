from collections.abc import Callable
from typing import Any, TypeVar

from torch.utils._pytree import PyTree, TreeSpec

type FlattenFuncSpec = Callable[[PyTree, TreeSpec], list]
type FlattenFuncExactMatchSpec = Callable[[PyTree, TreeSpec], bool]
SUPPORTED_NODES: dict[type[Any], FlattenFuncSpec] = ...
SUPPORTED_NODES_EXACT_MATCH: dict[type[Any], FlattenFuncExactMatchSpec | None] = ...
_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")

def register_pytree_flatten_spec(
    cls: type[Any],
    flatten_fn_spec: FlattenFuncSpec,
    flatten_fn_exact_match_spec: FlattenFuncExactMatchSpec | None = ...,
) -> None: ...
def tree_flatten_spec(pytree: PyTree, spec: TreeSpec) -> list[Any]: ...
