from collections.abc import Callable
from typing_extensions import deprecated

from torch.types import _TensorOrTensors

import torch

__all__ = [
    "GradcheckError",
    "get_analytical_jacobian",
    "get_numerical_jacobian",
    "get_numerical_jacobian_wrt_specific_input",
    "gradcheck",
    "gradgradcheck",
]

class GradcheckError(RuntimeError): ...

@deprecated(
    "`get_numerical_jacobian` was part of PyTorch's private API and not meant to be exposed. We are deprecating it and it will be removed in a future version of PyTorch. If you have a specific use for this or feature request for this to be a stable API, please file us an issue at https://github.com/pytorch/pytorch/issues/new",
    category=FutureWarning,
)
def get_numerical_jacobian(
    fn, inputs, target=..., eps=..., grad_out=...
) -> tuple[Tensor, ...]: ...
def get_numerical_jacobian_wrt_specific_input(
    fn, input_idx, inputs, outputs, eps, input=..., is_forward_ad=...
) -> tuple[torch.Tensor, ...]: ...

FAILED_NONDET_MSG = ...

@deprecated(
    "`get_analytical_jacobian` was part of PyTorch's private API and not meant to be exposed. We are deprecating it and it will be removed in a future version of PyTorch. If you have a specific use for this or feature request for this to be a stable API, please file us an issue at https://github.com/pytorch/pytorch/issues/new",
    category=FutureWarning,
)
def get_analytical_jacobian(
    inputs, output, nondet_tol=..., grad_out=...
) -> tuple[tuple[Tensor, ...], bool, bool, bool]: ...

FAILED_BATCHED_GRAD_MSG = ...
FAILED_BATCHED_GRAD_MSG_FWD_AD = ...
FAST_FAIL_SLOW_OK_MSG = ...

def gradcheck(
    func: Callable[..., _TensorOrTensors],
    inputs: _TensorOrTensors,
    *,
    eps: float = ...,
    atol: float = ...,
    rtol: float = ...,
    raise_exception: bool = ...,
    nondet_tol: float = ...,
    check_undefined_grad: bool = ...,
    check_grad_dtypes: bool = ...,
    check_batched_grad: bool = ...,
    check_batched_forward_grad: bool = ...,
    check_forward_ad: bool = ...,
    check_backward_ad: bool = ...,
    fast_mode: bool = ...,
    masked: bool | None = ...,
) -> bool: ...
def gradgradcheck(
    func: Callable[..., _TensorOrTensors],
    inputs: _TensorOrTensors,
    grad_outputs: _TensorOrTensors | None = ...,
    *,
    eps: float = ...,
    atol: float = ...,
    rtol: float = ...,
    gen_non_contig_grad_outputs: bool = ...,
    raise_exception: bool = ...,
    nondet_tol: float = ...,
    check_undefined_grad: bool = ...,
    check_grad_dtypes: bool = ...,
    check_batched_grad: bool = ...,
    check_fwd_over_rev: bool = ...,
    check_rev_over_rev: bool = ...,
    fast_mode: bool = ...,
    masked: bool = ...,
) -> bool: ...
