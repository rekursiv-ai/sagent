from torch._functorch.apis import (
    grad as grad,
    grad_and_value as grad_and_value,
    vmap as vmap,
)
from torch._functorch.batch_norm_replacement import (
    replace_all_batch_norm_modules_ as replace_all_batch_norm_modules_,
)
from torch._functorch.eager_transforms import (
    debug_unwrap as debug_unwrap,
    functionalize as functionalize,
    hessian as hessian,
    jacfwd as jacfwd,
    jacrev as jacrev,
    jvp as jvp,
    linearize as linearize,
    vjp as vjp,
)
from torch._functorch.functional_call import (
    functional_call as functional_call,
    stack_module_state as stack_module_state,
)

__all__ = [
    "debug_unwrap",
    "functional_call",
    "functionalize",
    "grad",
    "grad_and_value",
    "hessian",
    "jacfwd",
    "jacrev",
    "jvp",
    "linearize",
    "replace_all_batch_norm_modules_",
    "stack_module_state",
    "vjp",
    "vmap",
]
