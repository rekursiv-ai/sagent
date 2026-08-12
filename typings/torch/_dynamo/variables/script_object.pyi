from typing import Any
from .base import VariableTracker
from .user_defined import UserDefinedObjectVariable

"""
This module implements variable tracking for TorchScript objects during Dynamo tracing.

The TorchScriptObjectVariable class provides specialized handling for TorchScript
objects with strong safety guarantees by:
- Enforcing method-call-only access to prevent unsafe attribute manipulation
- Converting graph breaks into hard errors via _raise_hard_error_if_graph_break
- Proper proxy and source tracking for TorchScript method calls
- Integration with higher-order operators for method call handling

Key safety features:
- Strict validation that only method calls are allowed (no direct attribute access)
- Immediate error reporting for potentially unsafe operations
- Proper source tracking for debugging and guard installation
- Safe handling of TorchScript object method calls through torchbind

The module ensures that TorchScript objects are handled safely during tracing
by limiting operations to known-safe patterns and failing fast for unsafe usage.
"""

class TorchScriptObjectVariable(UserDefinedObjectVariable):
    _fake_script_object_cache: dict[int, TorchScriptObjectVariable] = ...
    @classmethod
    def is_matching_cls(cls, user_cls: type) -> bool: ...
    @staticmethod
    def create(proxy, value, **options) -> TorchScriptObjectVariable: ...
    def __init__(self, proxy, value, source, **kwargs) -> None: ...
    def as_proxy(self) -> Any: ...
    def var_getattr(self, tx, name: str) -> VariableTracker: ...
    def call_method(self, tx, name, args, kwargs): ...
