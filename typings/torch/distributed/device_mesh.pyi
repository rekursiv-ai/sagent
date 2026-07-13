from typing import Self

import types

from numpy.typing import ArrayLike as ArrayLike
from torch._C._distributed_c10d import Backend as C10dBackend
from torch.distributed.distributed_c10d import ProcessGroup

import torch

__all__ = ["DeviceMesh", "init_device_mesh"]

class DeviceMesh:
    device_type: str
    mesh: torch.Tensor
    mesh_dim_names: tuple[str, ...] | None
    def __init__(
        self,
        device_type: str,
        mesh: torch.Tensor | ArrayLike,
        *,
        mesh_dim_names: tuple[str, ...] | None = ...,
        backend_override: tuple[tuple[str | None, C10dBackend.Options | None], ...]
        | None = ...,
        _init_backend: bool = ...,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: types.TracebackType | None,
    ) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __getitem__(self, mesh_dim_names: str | tuple[str, ...]) -> DeviceMesh: ...
    def get_group(self, mesh_dim: int | str | None = ...) -> ProcessGroup: ...
    def get_all_groups(self) -> list[ProcessGroup]: ...
    @staticmethod
    def from_group(
        group: ProcessGroup | list[ProcessGroup],
        device_type: str,
        mesh: torch.Tensor | ArrayLike | None = ...,
        *,
        mesh_dim_names: tuple[str, ...] | None = ...,
    ) -> DeviceMesh: ...
    def size(self, mesh_dim: int | None = ...) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def get_rank(self) -> int: ...
    def get_local_rank(self, mesh_dim: int | str | None = ...) -> int: ...
    def get_coordinate(self) -> list[int] | None: ...

def init_device_mesh(
    device_type: str,
    mesh_shape: tuple[int, ...],
    *,
    mesh_dim_names: tuple[str, ...] | None = ...,
    backend_override: dict[
        int | str, str | C10dBackend.Options | tuple[str, C10dBackend.Options]
    ]
    | None = ...,
) -> DeviceMesh: ...
