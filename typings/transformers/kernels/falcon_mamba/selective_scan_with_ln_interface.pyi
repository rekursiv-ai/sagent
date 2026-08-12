from torch.cuda.amp import custom_bwd, custom_fwd

import mamba_ssm
import torch

if hasattr(mamba_ssm.ops.triton, "layernorm"): ...
else: ...

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        delta,
        A,
        B,
        C,
        D=...,
        z=...,
        delta_bias=...,
        delta_softplus=...,
        return_last_state=...,
    ):  # -> tuple[Any, Any]:
        ...
    @staticmethod
    def backward(
        ctx, dout, *args
    ):  # -> tuple[Any, Any, Any, Any, Any, Any | None, Any | None, Any | None, None, None]:
        ...

def rms_norm_forward(x, weight, bias, eps=..., is_rms_norm=...): ...
def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=...,
    z=...,
    delta_bias=...,
    delta_softplus=...,
    return_last_state=...,
):  # -> None:
    ...
def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=...,
    z=...,
    delta_bias=...,
    delta_softplus=...,
    return_last_state=...,
):  # -> Tensor | tuple[Tensor | Any, Any | None]:
    ...

class MambaInnerFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        out_proj_weight,
        out_proj_bias,
        A,
        B=...,
        C=...,
        D=...,
        delta_bias=...,
        B_proj_bias=...,
        C_proj_bias=...,
        delta_softplus=...,
        checkpoint_lvl=...,
        b_rms_weight=...,
        c_rms_weight=...,
        dt_rms_weight=...,
        b_c_dt_rms_eps=...,
    ):  # -> Tensor:
        ...
    @staticmethod
    @custom_bwd
    def backward(
        ctx, dout
    ):  # -> tuple[Tensor, Any, Any | None, Tensor, Tensor, Tensor, Any | None, Any, Any | None, Any | None, Any | None, Any | None, Any | None, Any | None, None, None, None, None, None, None]:
        ...

def mamba_inner_fn(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=...,
    C=...,
    D=...,
    delta_bias=...,
    B_proj_bias=...,
    C_proj_bias=...,
    delta_softplus=...,
    checkpoint_lvl=...,
    b_rms_weight=...,
    c_rms_weight=...,
    dt_rms_weight=...,
    b_c_dt_rms_eps=...,
):  # -> None:
    ...
