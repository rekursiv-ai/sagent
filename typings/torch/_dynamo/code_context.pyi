from typing import Any

import types

"""
This module provides thread-safe code context management for TorchDynamo using weak references.

The CodeContextDict class maintains a mapping between Python code objects and their associated
context data, using weak references to automatically clean up entries when code objects are
garbage collected. This prevents memory leaks while allowing context data to be associated
with code objects throughout their lifecycle.

Key features:
- Thread-safe context storage and retrieval
- Automatic cleanup using weak references
- Safe context management for Python code objects
- Memory-leak prevention

Example usage:
    code_obj = compile('x = 1', '<string>', 'exec')

    # Store context
    context = code_context.get_context(code_obj)
    context['metadata'] = {'optimized': True}

    # Retrieve context
    if code_context.has_context(code_obj):
        ctx = code_context.get_context(code_obj)
        # Use context data...

    # Remove context
    ctx = code_context.pop_context(code_obj)
"""

class CodeContextDict:
    def __init__(self) -> None: ...
    def has_context(self, code: types.CodeType) -> bool: ...
    def get_context(self, code: types.CodeType) -> dict[str, Any]: ...
    def pop_context(self, code: types.CodeType) -> dict[str, Any]: ...
    def clear(self) -> None: ...

code_context: CodeContextDict = ...
