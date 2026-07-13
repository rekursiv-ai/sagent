import sys

from ..decorators import substitute_in_graph

"""
Python polyfills for sys
"""
__all__ = ["getrecursionlimit", "intern"]

@substitute_in_graph(sys.intern, can_constant_fold_through=True)
def intern(string: str, /) -> str: ...
@substitute_in_graph(sys.getrecursionlimit, can_constant_fold_through=True)
def getrecursionlimit() -> int: ...

if hasattr(sys, "get_int_max_str_digits"):
    @substitute_in_graph(sys.get_int_max_str_digits, can_constant_fold_through=True)
    def get_int_max_str_digits() -> int: ...
