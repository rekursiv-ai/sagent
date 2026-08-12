from typing import Self
from collections.abc import Callable, Sequence
from typing import Any

from torch._library.custom_ops import CustomOpDef
from torch.overrides import TorchFunctionMode

import torch

def dontGenerateOpCheckTests(reason: str) -> Callable[..., Any]: ...
def is_abstract(tensor: torch.Tensor) -> bool: ...
def safe_schema_check(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    copy_inputs: bool = ...,
    rtol: float | None = ...,
    atol: float | None = ...,
) -> Any: ...
def safe_autograd_registration_check(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    copy_inputs: bool = ...,
    rtol: float | None = ...,
    atol: float | None = ...,
) -> None: ...
def safe_fake_check(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    copy_inputs: bool = ...,
    rtol: float | None = ...,
    atol: float | None = ...,
) -> None: ...
def safe_aot_autograd_check(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    dynamic: bool,
    *,
    copy_inputs: bool = ...,
    rtol: float | None = ...,
    atol: float | None = ...,
) -> Any: ...
def deepcopy_tensors(inputs: Any) -> Any: ...

ALL_TEST_UTILS = ...
GDOC = ...
DEFAULT_TEST_UTILS = ...
DEPRECATED_DEFAULT_TEST_UTILS = ...

def generate_opcheck_tests(
    testcase: Any,
    namespaces: list[str],
    failures_dict_path: str | None = ...,
    additional_decorators: dict[str, Callable] | None = ...,
    test_utils: list[str] = ...,
) -> None: ...
def generate_tag_tests(testcase, failures_dict, additional_decorators) -> None: ...

TEST_OPTIONS = ...

def validate_failures_dict_formatting(failures_dict_path: str) -> None: ...
def validate_failures_dict_structure(
    failure_dict: FailuresDict, test_utils: list[str], testcase: Any
) -> None: ...
def should_update_failures_dict() -> bool: ...

_is_inside_opcheck_mode = ...

def is_inside_opcheck_mode() -> Any: ...

class OpCheckMode(TorchFunctionMode):
    def __init__(
        self,
        namespaces: list[str],
        test_util_name: str,
        test_util: Callable,
        failures_dict: FailuresDict,
        test_name: str,
        failures_dict_path: str,
    ) -> None: ...
    def maybe_raise_errors_on_exit(self) -> None: ...
    def __enter__(self, *args, **kwargs) -> Self: ...
    def __exit__(self, *args, **kwargs) -> None: ...
    def run_test_util(self, op, args, kwargs) -> None: ...
    def __torch_function__(self, func, types, args=..., kwargs=...) -> Any: ...

def should_print_better_repro() -> None: ...
def opcheck(
    op: torch._ops.OpOverload | torch._ops.OpOverloadPacket | CustomOpDef[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = ...,
    *,
    test_utils: str | Sequence[str] = ...,
    raise_exception: bool = ...,
    rtol: float | None = ...,
    atol: float | None = ...,
) -> dict[str, str]: ...

class OpCheckError(Exception): ...

def generate_repro(
    test: str,
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    save_data: bool,
    dry_run: bool = ...,
) -> str: ...
def resolve_unique_overload_or_throw(
    op: torch._ops.OpOverloadPacket,
) -> torch._ops.OpOverload: ...

DUMP_OPTIONS = ...
type FailuresDictData = dict[str, dict[str, dict[str, str]]]
VERSION = ...
DESCRIPTION = ...

class FailuresDict:
    def __init__(self, path: str, data: FailuresDictData) -> None: ...
    @staticmethod
    def load(path, *, create_file=...) -> FailuresDict: ...
    def save(self) -> None: ...
    def get_status(self, qualname: str, test_name: str) -> str: ...
    def set_status(
        self, qualname: str, test_name: str, status: str, *, comment: str | None = ...
    ) -> None: ...
