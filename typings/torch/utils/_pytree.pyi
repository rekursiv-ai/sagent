from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    NamedTuple,
    NoReturn,
    Protocol,
    Self,
    TypeVar,
    overload,
)
from typing_extensions import deprecated

import dataclasses
import functools
import json
import types

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

This pytree implementation is not very performant due to Python overhead
To improve the performance we can move parts of the implementation to C++.
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
T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")
DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL = ...
NO_SERIALIZED_TYPE_NAME_FOUND = ...

class KeyEntry(Protocol):
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def get(self, parent: Any) -> Any: ...

class EnumEncoder(json.JSONEncoder):
    def default(self, obj: object) -> str | dict[str, Any]: ...

type Context = Any
type PyTree = Any
type FlattenFunc = Callable[[PyTree], tuple[list[Any], Context]]
type UnflattenFunc = Callable[[Iterable[Any], Context], PyTree]
type DumpableContext = Any
type ToDumpableContextFn = Callable[[Context], DumpableContext]
type FromDumpableContextFn = Callable[[DumpableContext], Context]
type ToStrFunc = Callable[[TreeSpec, list[str]], str]
type MaybeFromStrFunc = Callable[[str], tuple[Any, Context, str] | None]
type KeyPath = tuple[KeyEntry, ...]
type FlattenWithKeysFunc = Callable[[PyTree], tuple[list[tuple[KeyEntry, Any]], Any]]

class NodeDef(NamedTuple):
    type: type[Any]
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc
    flatten_with_keys_fn: FlattenWithKeysFunc | None

_NODE_REGISTRY_LOCK = ...
SUPPORTED_NODES: dict[type[Any], NodeDef] = ...

class _SerializeNodeDef(NamedTuple):
    typ: type[Any]
    serialized_type_name: str
    to_dumpable_context: ToDumpableContextFn | None
    from_dumpable_context: FromDumpableContextFn | None

SUPPORTED_SERIALIZED_TYPES: dict[type[Any], _SerializeNodeDef] = ...
SERIALIZED_TYPE_TO_PYTHON_TYPE: dict[str, type[Any]] = ...
_optree_minimum_version = ...
_optree_version = ...
_cxx_pytree_imported = ...
_cxx_pytree_pending_imports: list[Any] = ...

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
def register_dataclass(
    cls: type[Any],
    *,
    field_names: list[str] | None = ...,
    drop_field_names: list[str] | None = ...,
    serialized_type_name: str | None = ...,
) -> None: ...

CONSTANT_NODES: set[type] = ...

def register_constant(cls: type[Any]) -> None: ...
def is_constant_class(cls: type[Any]) -> bool: ...

@dataclasses.dataclass(frozen=True)
class ConstantNode:
    value: Any

@dataclasses.dataclass(frozen=True)
class SequenceKey(Generic[T]):
    idx: int

    def get(self, sequence: Sequence[T]) -> T: ...

K = TypeVar("K", bound=Hashable)

@dataclasses.dataclass(frozen=True)
class MappingKey(Generic[K, T]):
    key: K

    def get(self, mapping: Mapping[K, T]) -> T: ...

@dataclasses.dataclass(frozen=True)
class GetAttrKey:
    name: str

    def get(self, obj: Any) -> Any: ...

def is_namedtuple(obj: object | type) -> bool: ...
def is_namedtuple_class(cls: type) -> bool: ...
def is_namedtuple_instance(obj: object) -> bool: ...

_T_co = TypeVar("_T_co", covariant=True)

class structseq(tuple[_T_co, ...]):
    __slots__: ClassVar[tuple[()]] = ...
    n_fields: Final[int]
    n_sequence_fields: Final[int]
    n_unnamed_fields: Final[int]
    def __init_subclass__(cls) -> NoReturn: ...
    def __new__(
        cls: type[Self], sequence: Iterable[_T_co], dict: dict[str, Any] = ...
    ) -> Self: ...

def is_structseq(obj: object | type) -> bool: ...

Py_TPFLAGS_BASETYPE: int = ...

def is_structseq_class(cls: type) -> bool: ...
def is_structseq_instance(obj: object) -> bool: ...

_odict_flatten = ...
_odict_unflatten = ...
STANDARD_DICT_TYPES: frozenset[type] = ...
BUILTIN_TYPES: frozenset[type] = ...

def tree_is_leaf(
    tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = ...
) -> bool: ...

@dataclasses.dataclass(init=True, frozen=True, eq=True, repr=False)
class TreeSpec:
    type: Any
    context: Context
    children_specs: list[TreeSpec]
    num_nodes: int = ...
    num_leaves: int = ...
    num_children: int = ...
    def __post_init__(self) -> None: ...
    def __repr__(self, indent: int = ...) -> str: ...
    def __eq__(self, other: PyTree) -> bool: ...
    def is_leaf(self) -> bool: ...
    def flatten_up_to(self, tree: PyTree) -> list[PyTree]: ...
    def unflatten(self, leaves: Iterable[Any]) -> PyTree: ...
    def __hash__(self) -> int: ...

@dataclasses.dataclass(init=True, frozen=True, eq=False, repr=False)
class LeafSpec(TreeSpec):
    type: Any = ...
    context: Context = ...
    children_specs: list[TreeSpec] = ...
    def __post_init__(self) -> None: ...
    def __repr__(self, indent: int = ...) -> str: ...

_LEAF_SPEC = ...

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

@dataclasses.dataclass
class _TreeSpecSchema:
    type: str | None
    context: DumpableContext
    children_spec: list[_TreeSpecSchema]

class _ProtocolFn(NamedTuple):
    treespec_to_json: Callable[[TreeSpec], DumpableContext]
    json_to_treespec: Callable[[DumpableContext], TreeSpec]

_SUPPORTED_PROTOCOLS: dict[int, _ProtocolFn] = ...

def enum_object_hook(obj: dict[str, Any]) -> Enum | dict[str, Any]: ...
def treespec_dumps(treespec: TreeSpec, protocol: int | None = ...) -> str: ...
@functools.lru_cache
def treespec_loads(serialized: str) -> TreeSpec: ...

class _DummyLeaf: ...

def treespec_pprint(treespec: TreeSpec) -> str: ...
@deprecated(
    "`pytree_to_str` is deprecated. Please use `treespec_dumps` instead.",
    category=FutureWarning,
)
def pytree_to_str(treespec: TreeSpec) -> str: ...
@deprecated(
    "`str_to_pytree` is deprecated. Please use `treespec_loads` instead.",
    category=FutureWarning,
)
def str_to_pytree(json: str) -> TreeSpec: ...
def arg_tree_leaves(*args: PyTree, **kwargs: PyTree) -> list[Any]: ...
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
