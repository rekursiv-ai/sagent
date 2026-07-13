from . import (
    compiled_autograd as compiled_autograd,
    eval_frame as eval_frame,
    guards as guards,
)

def strip_function_call(name: str) -> str: ...
def is_valid_var_name(name: str) -> bool | int: ...
