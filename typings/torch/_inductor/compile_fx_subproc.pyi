from typing import final

import functools

from torch._inductor.compile_worker.subproc_pool import AnyPool

from .compile_fx_ext import _OutOfProcessFxCompile

@final
class _SubprocessFxCompile(_OutOfProcessFxCompile):
    @staticmethod
    @functools.cache
    def process_pool() -> AnyPool: ...
