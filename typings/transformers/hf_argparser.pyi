from argparse import ArgumentParser
from collections.abc import Callable, Iterable
from typing import Any, NewType

import dataclasses
import os

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)

def string_to_bool(v):  # -> bool:
    ...
def make_choice_type_function(choices: list) -> Callable[[str], Any]: ...
def HfArg(
    *,
    aliases: str | list[str] | None = ...,
    help: str | None = ...,
    default: Any = ...,
    default_factory: Callable[[], Any] = ...,
    metadata: dict | None = ...,
    **kwargs,
) -> dataclasses.Field: ...

class HfArgumentParser(ArgumentParser):
    dataclass_types: Iterable[DataClassType]
    def __init__(
        self,
        dataclass_types: DataClassType | Iterable[DataClassType] | None = ...,
        **kwargs,
    ) -> None: ...
    def parse_args_into_dataclasses(
        self,
        args=...,
        return_remaining_strings=...,
        look_for_args_file=...,
        args_filename=...,
        args_file_flag=...,
    ) -> tuple[DataClass, ...]: ...
    def parse_dict(
        self, args: dict[str, Any], allow_extra_keys: bool = ...
    ) -> tuple[DataClass, ...]: ...
    def parse_json_file(
        self, json_file: str | os.PathLike, allow_extra_keys: bool = ...
    ) -> tuple[DataClass, ...]: ...
    def parse_yaml_file(
        self, yaml_file: str | os.PathLike, allow_extra_keys: bool = ...
    ) -> tuple[DataClass, ...]: ...
