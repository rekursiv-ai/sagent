from typing import TypeVar

from torch import Tensor

T = TypeVar("T")
type _scalar_or_tuple_any_t[T] = T | tuple[T, ...]
type _scalar_or_tuple_1_t[T] = T | tuple[T]
type _scalar_or_tuple_2_t[T] = T | tuple[T, T]
type _scalar_or_tuple_3_t[T] = T | tuple[T, T, T]
type _scalar_or_tuple_4_t[T] = T | tuple[T, T, T, T]
type _scalar_or_tuple_5_t[T] = T | tuple[T, T, T, T, T]
type _scalar_or_tuple_6_t[T] = T | tuple[T, T, T, T, T, T]
type _size_any_t = _scalar_or_tuple_any_t[int]
type _size_1_t = _scalar_or_tuple_1_t[int]
type _size_2_t = _scalar_or_tuple_2_t[int]
type _size_3_t = _scalar_or_tuple_3_t[int]
type _size_4_t = _scalar_or_tuple_4_t[int]
type _size_5_t = _scalar_or_tuple_5_t[int]
type _size_6_t = _scalar_or_tuple_6_t[int]
type _size_any_opt_t = _scalar_or_tuple_any_t[int | None]
type _size_2_opt_t = _scalar_or_tuple_2_t[int | None]
type _size_3_opt_t = _scalar_or_tuple_3_t[int | None]
type _ratio_2_t = _scalar_or_tuple_2_t[float]
type _ratio_3_t = _scalar_or_tuple_3_t[float]
type _ratio_any_t = _scalar_or_tuple_any_t[float]
type _tensor_list_t = _scalar_or_tuple_any_t[Tensor]
type _maybe_indices_t = _scalar_or_tuple_2_t[Tensor]
