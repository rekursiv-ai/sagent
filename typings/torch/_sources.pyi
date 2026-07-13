from typing import Any, NamedTuple

import ast
import functools

from torch._C._jit_tree_views import SourceRangeFactory

def get_source_lines_and_file(
    obj: Any, error_msg: str | None = ...
) -> tuple[list[str], int, str | None]: ...
def normalize_source_lines(sourcelines: list[str]) -> list[str]: ...

class SourceContext(SourceRangeFactory):
    def __init__(
        self,
        source,
        filename,
        file_lineno,
        leading_whitespace_len,
        uses_true_division=...,
        funcname=...,
    ) -> None: ...

@functools.cache
def make_source_context(*args) -> SourceContext: ...
def fake_range() -> SourceRange: ...

class ParsedDef(NamedTuple):
    ast: ast.Module
    ctx: SourceContext
    source: str
    filename: str | None
    file_lineno: int

def parse_def(fn) -> ParsedDef: ...
