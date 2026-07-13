from typing import AnyStr

import os

from ..decorators import substitute_in_graph

"""
Python polyfills for os
"""
__all__ = ["fspath"]

@substitute_in_graph(os.fspath, can_constant_fold_through=True)
def fspath(path: AnyStr | os.PathLike[AnyStr]) -> AnyStr: ...
