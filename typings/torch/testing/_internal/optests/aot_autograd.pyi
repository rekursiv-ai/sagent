from typing import Literal

class assert_raises_regex:
    def __init__(self, exception_cls, regex) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_val, traceback) -> Literal[True]: ...

def aot_autograd_check(
    func,
    args,
    kwargs,
    dynamic,
    assert_raises_regex_fn=...,
    assert_equals_fn=...,
    check_gradients=...,
    try_check_data_specialization=...,
    skip_correctness_check=...,
) -> None: ...

outputs_msg = ...
