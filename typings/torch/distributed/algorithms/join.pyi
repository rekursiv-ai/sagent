from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import types

import torch

__all__ = ["Join", "JoinHook", "Joinable"]

class JoinHook:
    def main_hook(self) -> None: ...
    def post_hook(self, is_last_joiner: bool) -> None: ...

class Joinable(ABC):
    @abstractmethod
    def __init__(self) -> None: ...
    @abstractmethod
    def join_hook(self, **kwargs) -> JoinHook: ...
    @property
    @abstractmethod
    def join_device(self) -> torch.device: ...
    @property
    @abstractmethod
    def join_process_group(self) -> Any: ...

class _JoinConfig(NamedTuple):
    enable: bool
    throw_on_early_termination: bool
    is_first_joinable: bool
    @staticmethod
    def construct_disabled_join_config() -> _JoinConfig: ...

class Join:
    def __init__(
        self,
        joinables: list[Joinable],
        enable: bool = ...,
        throw_on_early_termination: bool = ...,
        **kwargs,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    @staticmethod
    def notify_join_context(joinable: Joinable) -> Any | _IllegalWork | Work | None: ...
