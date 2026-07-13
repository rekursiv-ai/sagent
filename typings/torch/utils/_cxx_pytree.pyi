from collections.abc import Callable, Iterable
from typing import Any, Self, TypeVar, overload

import functools
import types

from optree import PyTreeSpec as TreeSpec
from torch.utils._pytree import KeyEntry as KeyEntry

"""
Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_leaves` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.
"""
__all__ = [
    "Context",
    "DumpableContext",
    "FlattenFunc",
    "FromDumpableContextFn",
    "LeafSpec",
    "PyTree",
    "ToDumpableContextFn",
    "TreeSpec",
    "UnflattenFunc",
    "is_namedtuple",
    "is_namedtuple_class",
    "is_namedtuple_instance",
    "is_structseq",
    "is_structseq_class",
    "is_structseq_instance",
    "key_get",
    "keystr",
    "register_pytree_node",
    "tree_all",
    "tree_all_only",
    "tree_any",
    "tree_any_only",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_is_leaf",
    "tree_iter",
    "tree_leaves",
    "tree_leaves_with_path",
    "tree_map",
    "tree_map_",
    "tree_map_only",
    "tree_map_only_",
    "tree_map_with_path",
    "tree_structure",
    "tree_unflatten",
    "treespec_dumps",
    "treespec_loads",
    "treespec_pprint",
]
__TORCH_DICT_SESSION = ...
T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")
type Context = Any
type PyTree = Any
type FlattenFunc = Callable[[PyTree], tuple[list[Any], Context]]
type UnflattenFunc = Callable[[Iterable[Any], Context], PyTree]
type OpTreeUnflattenFunc = Callable[[Context, Iterable[Any]], PyTree]
type DumpableContext = Any
type ToDumpableContextFn = Callable[[Context], DumpableContext]
type FromDumpableContextFn = Callable[[DumpableContext], Context]
type KeyPath = tuple[KeyEntry, ...]
type FlattenWithKeysFunc = Callable[[PyTree], tuple[list[tuple[KeyEntry, Any]], Any]]

def register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: str | None = ...,
    to_dumpable_context: ToDumpableContextFn | None = ...,
    from_dumpable_context: FromDumpableContextFn | None = ...,
    flatten_with_keys_fn: FlattenWithKeysFunc | None = ...,
) -> None: ...
def tree_is_leaf(
    tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = ...
) -> bool: ...
def tree_flatten(
    tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = ...
) -> tuple[list[Any], TreeSpec]: ...
def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree: ...
def tree_iter(
    tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = ...
) -> Iterable[Any]: ...
def tree_leaves(
    tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = ...
) -> list[Any]: ...
def tree_structure(
    tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = ...
) -> TreeSpec: ...
def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
def tree_map_(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...

type Type2[T, S] = tuple[type[T], type[S]]
type Type3[T, S, U] = tuple[type[T], type[S], type[U]]
type TypeAny = type[Any] | tuple[type[Any], ...] | types.UnionType
type Fn2[T, S, R] = Callable[[T | S], R]
type Fn3[T, S, U, R] = Callable[[T | S | U], R]
type Fn[T, R] = Callable[[T], R]
type FnAny[R] = Callable[[Any], R]
type MapOnlyFn[T] = Callable[[T], Callable[[Any], Any]]

@overload
def map_only(type_or_types_or_pred: type[T], /) -> MapOnlyFn[Fn[T, Any]]: ...
@overload
def map_only(type_or_types_or_pred: Type2[T, S], /) -> MapOnlyFn[Fn2[T, S, Any]]: ...
@overload
def map_only(
    type_or_types_or_pred: Type3[T, S, U], /
) -> MapOnlyFn[Fn3[T, S, U, Any]]: ...
@overload
def map_only(type_or_types_or_pred: TypeAny, /) -> MapOnlyFn[FnAny[Any]]: ...
@overload
def map_only(
    type_or_types_or_pred: Callable[[Any], bool], /
) -> MapOnlyFn[FnAny[Any]]: ...
def map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool], /
) -> MapOnlyFn[FnAny[Any]]: ...
@overload
def tree_map_only(
    type_or_types_or_pred: type[T],
    /,
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
@overload
def tree_map_only(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
@overload
def tree_map_only(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
@overload
def tree_map_only(
    type_or_types_or_pred: TypeAny,
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
@overload
def tree_map_only(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
def tree_map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: type[T],
    /,
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: TypeAny,
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
def tree_map_only_(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
def tree_all(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
def tree_any(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
@overload
def tree_all_only(
    type_or_types: type[T],
    /,
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
@overload
def tree_all_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
@overload
def tree_all_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
def tree_all_only(
    type_or_types: TypeAny,
    /,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
@overload
def tree_any_only(
    type_or_types: type[T],
    /,
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
@overload
def tree_any_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
@overload
def tree_any_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
def tree_any_only(
    type_or_types: TypeAny,
    /,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> bool: ...
def broadcast_prefix(
    prefix_tree: PyTree,
    full_tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> list[Any]: ...
def treespec_dumps(treespec: TreeSpec, protocol: int | None = ...) -> str: ...
@functools.lru_cache
def treespec_loads(serialized: str) -> TreeSpec: ...

class _DummyLeaf: ...

def treespec_pprint(treespec: TreeSpec) -> str: ...

class LeafSpecMeta(type(TreeSpec)):
    def __instancecheck__(self, instance: object) -> bool: ...

class LeafSpec(TreeSpec, metaclass=LeafSpecMeta):
    def __new__(cls) -> Self: ...

def tree_flatten_with_path(
    tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = ...
) -> tuple[list[tuple[KeyPath, Any]], TreeSpec]: ...
def tree_leaves_with_path(
    tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = ...
) -> list[tuple[KeyPath, Any]]: ...
def tree_map_with_path(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = ...,
) -> PyTree: ...
def keystr(kp: KeyPath) -> str: ...
def key_get(obj: Any, kp: KeyPath) -> Any: ...
