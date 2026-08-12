from typing import Any, Literal
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TypeVar
from typing_extensions import ParamSpec

from torch import Tensor
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import out_wrapper

import torch

_T = TypeVar("_T")
_P = ParamSpec("_P")
aten = ...
_meta_lib_dont_use_me_use_register_meta = ...

def register_meta(op) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def elementwise_meta(
    *args, type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND
) -> FakeTensor: ...
def toRealValueType(dtype) -> dtype | None: ...
def check_inplace_broadcast(self_shape, *args_shape) -> None: ...
@register_meta([aten.linspace, aten.logspace])
@out_wrapper()
def meta_linspace_logspace(
    start,
    end,
    steps,
    base=...,
    dtype=...,
    device=...,
    layout=...,
    pin_memory=...,
    requires_grad=...,
) -> Tensor: ...
@register_meta([aten.take.default, aten.take.out])
@out_wrapper()
def meta_take(self, index): ...
@register_meta([aten.linalg_cross.default, aten.linalg_cross.out])
@out_wrapper()
def linalg_cross(self, other, *, dim=...): ...
@register_meta(aten.linalg_matrix_exp)
@out_wrapper()
def linalg_matrix_exp(self) -> Tensor: ...
@register_meta(
    [aten.cummax.default, aten.cummax.out, aten.cummin.default, aten.cummin.out]
)
@out_wrapper("values", "indices")
def cummaxmin(self, dim) -> tuple[Tensor, Tensor]: ...
@register_meta([aten.logcumsumexp.default, aten.logcumsumexp.out])
@out_wrapper()
def logcumsumexp(self, dim) -> Tensor: ...
@register_meta([aten._fft_c2c.default, aten._fft_c2c.out])
@out_wrapper()
def meta_fft_c2c(self, dim, normalization, forward): ...

cufft_max_ndim = ...

def use_optimized_cufft_path(dim: list[int]) -> bool: ...
@register_meta([aten._fft_r2c.default, aten._fft_r2c.out])
@out_wrapper()
def meta_fft_r2c(self, dim, normalization, onesided): ...
@register_meta(aten.randperm.generator_out)
def meta_randperm(n, *, generator=..., out) -> Tensor: ...
@register_meta(aten.randperm.default)
def meta_randperm_default(
    n, *, dtype=..., layout=..., device=..., pin_memory=...
) -> Tensor: ...
@register_meta([aten.randint.default, aten.randint.out])
@out_wrapper()
def meta_randint(
    high, size, *, dtype=..., layout=..., device=..., pin_memory=...
) -> Tensor: ...
@register_meta([aten.randint.low, aten.randint.low_out])
@out_wrapper()
def meta_randint_low(
    low, high, size, *, dtype=..., layout=..., device=..., pin_memory=...
) -> Tensor: ...
@register_meta([aten.rand.default, aten.rand.out])
@out_wrapper()
def meta_rand_default(
    size, *, dtype=..., layout=..., device=..., pin_memory=...
) -> Tensor: ...
@register_meta([aten._fft_c2r.default, aten._fft_c2r.out])
@out_wrapper()
def meta_fft_c2r(
    self: Tensor, dim: list[int], normalization: int, lastdim: int
) -> Tensor: ...
@register_meta(aten.copy_.default)
def meta_copy_(self, src, non_blocking=...): ...
def inferUnsqueezeGeometry(tensor, dim) -> tuple[list[Any], list[Any]]: ...
@register_meta(aten.unsqueeze_.default)
def meta_unsqueeze_(self, dim): ...
@register_meta(aten._sparse_semi_structured_linear)
def meta_sparse_structured_linear(
    input: Tensor,
    weight: Tensor,
    _meta: Tensor,
    bias: Tensor | None = ...,
    _activation_opt: str | None = ...,
    out_dtype: torch.dtype | None = ...,
) -> Tensor: ...
@register_meta(aten._sparse_semi_structured_mm)
def meta_sparse_structured_mm(
    mat1: Tensor, mat1_meta: Tensor, mat2: Tensor, out_dtype: torch.dtype | None = ...
) -> Tensor: ...
@register_meta(aten._sparse_semi_structured_addmm)
def meta_sparse_structured_addmm(
    input: Tensor,
    mat1: Tensor,
    mat1_meta: Tensor,
    mat2: Tensor,
    *,
    alpha=...,
    beta=...,
    out_dtype: torch.dtype | None = ...,
) -> Tensor: ...
@register_meta(aten._cslt_sparse_mm)
def meta__cslt_sparse_mm(
    compressed_A: torch.Tensor,
    dense_B: torch.Tensor,
    bias: Tensor | None = ...,
    alpha: Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
    transpose_result: bool = ...,
    alg_id: int = ...,
    split_k: int = ...,
    split_k_mode: int = ...,
) -> Tensor: ...
@register_meta(aten.index_reduce.default)
def meta_index_reduce(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = ...,
) -> Tensor: ...
@register_meta(aten.index_reduce_.default)
def meta_index_reduce_(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = ...,
) -> Tensor: ...
@out_wrapper()
@register_meta(aten.index_select.default)
def meta_index_select(self, dim, index): ...
@register_meta(aten.segment_reduce.default)
def meta_segment_reduce(
    data: Tensor,
    reduce: str,
    *,
    lengths: Tensor | None = ...,
    indices: Tensor | None = ...,
    offsets: Tensor | None = ...,
    axis: int = ...,
    unsafe: bool = ...,
    initial=...,
) -> Tensor: ...
@register_meta([aten.max.default, aten.max.unary_out])
@out_wrapper()
def meta_max(self): ...
@register_meta(aten.max.dim)
def meta_max_dim(self, dim, keepdim=...) -> tuple[Any, Any]: ...
@register_meta([aten.min.default, aten.min.unary_out])
@out_wrapper()
def meta_min(self): ...
@register_meta(aten.min.dim)
def meta_min_dim(self, dim, keepdim=...) -> tuple[Any, Any]: ...
@register_meta(aten.angle.default)
def meta_angle(self) -> Tensor: ...
@register_meta(aten.angle.out)
def meta_angle_out(self, out): ...
@register_meta(aten._assert_async.default)
def assert_async(val) -> None: ...
@register_meta(aten._assert_async.msg)
def assert_async_meta(val, assert_msg) -> None: ...
@register_meta(aten._print.default)
def print_meta(s) -> None: ...
@register_meta(aten._make_dep_token.default)
def make_dep_token(
    *, dtype=..., layout=..., device=..., pin_memory=..., memory_format=...
) -> Tensor: ...
@register_meta(aten.sym_constrain_range.default)
def sym_constrain_range(size, min=..., max=...) -> None: ...
@register_meta(aten._functional_sym_constrain_range.default)
def functional_sym_constrain_range(size, min=..., max=..., dep_token=...) -> None: ...
@register_meta(aten.sym_constrain_range_for_size.default)
def sym_constrain_range_for_size(size, min=..., max=...) -> None: ...
@register_meta(aten._functional_sym_constrain_range_for_size.default)
def functional_sym_constrain_range_for_size(size, min, max, dep_token): ...
@register_meta(aten._functional_assert_async.msg)
def functional_assert_async_meta(val, assert_msg, dep_token): ...
def squareCheckInputs(self: Tensor, f_name: str) -> None: ...
def linearSolveCheckInputs(self: Tensor, A: Tensor, name: str) -> None: ...
def checkFloatingOrComplex(
    t: Tensor, f_name: str, allow_low_precision_dtypes: bool = ...
) -> None: ...
def checkIsMatrix(A: Tensor, f_name: str, arg_name: str = ...) -> None: ...
def checkInputsSolver(A: Tensor, B: Tensor, left: bool, f_name: str) -> None: ...
def checkSameDevice(
    fn_name: str, result: Tensor, input: Tensor, result_name: str = ...
) -> None: ...
def checkUplo(UPLO: str) -> None: ...
@register_meta([aten._linalg_eigh.default, aten._linalg_eigh.eigenvalues])
@out_wrapper("eigenvalues", "eigenvectors")
def meta__linalg_eigh(
    A: Tensor, UPLO: str = ..., compute_v: bool = ...
) -> tuple[Tensor, Tensor]: ...
@register_meta([aten._linalg_eigvals.default, aten.linalg_eigvals.out])
@out_wrapper()
def meta__linalg_eigvals(input: Tensor) -> Tensor: ...
@register_meta([aten.linalg_eig])
@out_wrapper("eigenvalues", "eigenvectors")
def meta_linalg_eig(input: Tensor) -> tuple[Tensor, Tensor]: ...
def cloneBatchedColumnMajor(src: Tensor) -> Tensor: ...
@register_meta(aten.cholesky_solve)
@out_wrapper()
def cholesky_solve(self: Tensor, A: Tensor, upper: bool = ...) -> Tensor: ...
@register_meta(aten.cholesky)
@out_wrapper()
def cholesky(self: Tensor, upper: bool = ...) -> Tensor: ...
@register_meta(aten.cholesky_inverse)
@out_wrapper()
def cholesky_inverse(self: Tensor, upper: bool = ...) -> Tensor: ...
@register_meta(aten.linalg_cholesky_ex.default)
def linalg_cholesky_ex(
    A: Tensor, upper: bool = ..., check_errors: bool = ...
) -> tuple[Tensor, Tensor]: ...
@register_meta(
    [aten.linalg_householder_product.default, aten.linalg_householder_product.out]
)
@out_wrapper()
def linalg_householder_product(input: Tensor, tau: Tensor) -> Tensor: ...
@register_meta(aten.linalg_inv_ex.default)
def linalg_inv_ex_meta(
    A: Tensor, check_errors: bool = ...
) -> tuple[Tensor, Tensor]: ...
@register_meta([aten.linalg_ldl_factor_ex.default, aten.linalg_ldl_factor_ex.out])
@out_wrapper("LD", "pivots", "info")
def linalg_ldl_factor_ex_meta(
    self: Tensor, *, hermitian: bool = ..., check_errors: bool = ...
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_meta([aten.linalg_ldl_solve.default, aten.linalg_ldl_solve.out])
@out_wrapper()
def linalg_ldl_solve_meta(
    LD: Tensor, pivots: Tensor, B: Tensor, *, hermitian: bool = ...
) -> Tensor: ...
@register_meta([aten.linalg_lu.default, aten.linalg_lu.out])
@out_wrapper("P", "L", "U")
def linalg_lu_meta(
    A: Tensor, *, pivot: bool = ...
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_meta([aten.linalg_lu_factor_ex.default, aten.linalg_lu_factor_ex.out])
@out_wrapper("LU", "pivots", "info")
def linalg_lu_factor_ex_meta(
    A: Tensor, *, pivot: bool = ..., check_errors: bool = ...
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_meta([aten.linalg_lu_solve.default, aten.linalg_lu_solve.out])
@out_wrapper()
def linalg_lu_solve_meta(
    LU: Tensor, pivots: Tensor, B: Tensor, *, left: bool = ..., adjoint: bool = ...
) -> Tensor: ...
@register_meta(aten.lu_unpack)
@out_wrapper("P", "L", "U")
def lu_unpack_meta(
    LU: Tensor, pivots: Tensor, unpack_data: bool = ..., unpack_pivots: bool = ...
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_meta([aten.linalg_qr.default, aten.linalg_qr.out])
@out_wrapper("Q", "R")
def linalg_qr_meta(A: Tensor, mode: str = ...) -> tuple[Tensor, Tensor]: ...
def linalg_solve_is_vector_rhs(input: Tensor, other: Tensor) -> bool: ...
@register_meta([aten.linalg_solve_triangular.default, aten.linalg_solve_triangular.out])
def linalg_solve_triangular_meta(
    A: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = ...,
    unitriangular: bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@register_meta(aten.triangular_solve)
@out_wrapper("X", "M", exact_dtype=True)
def triangular_solve_meta(
    self: Tensor,
    A: Tensor,
    upper: bool = ...,
    transpose: bool = ...,
    unitriangular: bool = ...,
) -> tuple[Tensor, Tensor]: ...
@register_meta(aten.ormqr)
@out_wrapper()
def ormqr(
    input: Tensor, tau: Tensor, other: Tensor, left: bool = ..., transpose: bool = ...
) -> Tensor: ...
@register_meta(aten.reflection_pad1d)
@out_wrapper()
def meta_reflection_pad1d(input, padding): ...
@register_meta(aten.replication_pad1d)
@out_wrapper()
def meta_replication_pad1d(input, padding): ...
@register_meta(aten.reflection_pad1d_backward)
@out_wrapper("grad_input")
def meta_reflection_pad1d_backward(grad_output, input, padding): ...
@register_meta(aten.replication_pad1d_backward)
@out_wrapper("grad_input")
def meta_replication_pad1d_backward(grad_output, input, padding): ...
@register_meta(aten.reflection_pad2d)
@out_wrapper()
def meta_reflection_pad2d(input, padding): ...
@register_meta(aten.replication_pad2d)
@out_wrapper()
def meta_replication_pad2d(input, padding): ...
@register_meta(
    [
        aten.reflection_pad2d_backward.default,
        aten.reflection_pad2d_backward.grad_input,
        aten.replication_pad2d_backward.default,
        aten.replication_pad2d_backward.grad_input,
    ]
)
@out_wrapper("grad_input")
def meta_pad2d_backward(grad_output, self, padding): ...
@register_meta(aten.reflection_pad3d)
@out_wrapper()
def meta_reflection_pad3d(input, padding): ...
@register_meta(aten.replication_pad3d)
@out_wrapper()
def meta_replication_pad3d(input, padding): ...
@register_meta(
    [
        aten.reflection_pad3d_backward.default,
        aten.reflection_pad3d_backward.grad_input,
        aten.replication_pad3d_backward.default,
        aten.replication_pad3d_backward.grad_input,
    ]
)
@out_wrapper("grad_input")
def meta_pad3d_backward(grad_output, input, padding): ...
@register_meta(aten._pdist_forward)
@out_wrapper()
def meta__pdist_forward(self: Tensor, p: float = ...) -> Tensor: ...
@register_meta(aten._pdist_backward)
@out_wrapper()
def meta__pdist_backward(
    grad: Tensor, self: Tensor, p: float, pdist: Tensor
) -> Tensor: ...
@register_meta([aten.baddbmm.default, aten.baddbmm.out])
@out_wrapper(exact_dtype=True)
def meta_baddbmm(self, batch1, batch2, *, beta=..., alpha=...): ...
@register_meta([aten.bernoulli.default, aten.bernoulli.out])
@out_wrapper()
def meta_bernoulli(self, *, generator=...) -> Tensor: ...
@register_meta(aten.bernoulli_.float)
def meta_bernoulli_(self, p=..., generator=...): ...
@register_meta(aten.bernoulli.p)
def meta_bernoulli_p(self, p=..., generator=...) -> Tensor: ...
@register_meta([aten.poisson.default, aten.poisson.out])
@out_wrapper()
def meta_poisson(self, generator=...) -> Tensor: ...
@register_meta(aten._fused_moving_avg_obs_fq_helper.default)
def meta__fused_moving_avg_obs_fq_helper(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=...,
    symmetric_quant=...,
) -> tuple[Tensor, Tensor]: ...
@register_meta(aten.mm)
@out_wrapper(exact_dtype=True)
def meta_mm(a, b): ...
def device_hint(tensor) -> str: ...
def calc_conv_nd_return_shape(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int] | int,
    padding: list[int] | int,
    dilation: list[int] | int,
    is_transposed: bool,
    groups: int,
    output_padding: list[int] | int | None = ...,
) -> list[int]: ...
def is_channels_last(ten): ...
@register_meta(aten.miopen_batch_norm.default)
def meta_miopen_batch_norm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_meta(aten.convolution.default)
def meta_conv(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    is_transposed: bool,
    output_padding: list[int],
    groups: int,
) -> Tensor: ...

if torch._C._has_mkldnn:
    _meta_lib_dont_use_me_use_register_meta_for_mkldnn = ...
    @register_meta(torch.ops.mkldnn._convolution_pointwise.default)
    def meta_mkldnn_convolution_default(
        input_tensor,
        weight,
        bias,
        padding,
        stride,
        dilation,
        groups,
        attr,
        scalars,
        algorithm,
    ): ...
    @register_meta(torch.ops.mkldnn._linear_pointwise.default)
    def meta_linear_pointwise_default(
        input_tensor, weight, bias, attr, scalars, algorithm
    ): ...

    _meta_lib_dont_use_me_use_register_meta_for_onednn = ...
    @register_meta(torch.ops.onednn.qconv2d_pointwise.default)
    @register_meta(torch.ops.onednn.qconv_pointwise.default)
    def meta_qconv_pointwise(
        x,
        x_scale,
        x_zp,
        w,
        w_scale,
        w_zp,
        bias,
        stride,
        padding,
        dilation,
        groups,
        output_scale,
        output_zero_point,
        output_dtype,
        attr,
        scalars,
        algorithm,
    ): ...
    @register_meta(torch.ops.onednn.qconv2d_pointwise.binary)
    def meta_qconv2d_pointwise_binary(
        x,
        x_scale,
        x_zp,
        w,
        w_scale,
        w_zp,
        accum,
        bias,
        stride,
        padding,
        dilation,
        groups,
        output_scale,
        output_zero_point,
        output_dtype,
        accum_scale,
        accum_zero_point,
        binary_op_name,
        alpha,
        unary_op_name,
        unary_op_args,
        unary_op_algorithm,
    ): ...
    @register_meta(torch.ops.onednn.qlinear_pointwise.default)
    @register_meta(torch.ops.onednn.qlinear_pointwise.tensor)
    def meta_qlinear_pointwise(
        x,
        x_scale,
        x_zp,
        w,
        w_scale,
        w_zp,
        bias,
        output_scale,
        output_zero_point,
        output_dtype,
        post_op_name,
        post_op_args,
        post_op_algorithm,
    ): ...
    @register_meta(torch.ops.onednn.qlinear_pointwise.binary)
    @register_meta(torch.ops.onednn.qlinear_pointwise.binary_tensor)
    def meta_qlinear_pointwise_binary(
        x,
        x_scale,
        x_zp,
        w,
        w_scale,
        w_zp,
        x_2,
        bias,
        output_scale,
        output_zero_point,
        output_dtype,
        x2_scale,
        x2_zp,
        binary_op_name,
        alpha,
        unary_op_name,
        unary_op_args,
        unary_op_algorithm,
    ): ...
    @register_meta(torch.ops.onednn.linear_dynamic_fp16.default)
    @register_meta(torch.ops.onednn.linear_relu_dynamic_fp16.default)
    def meta_linear_dynamic_fp16(x, w, bias): ...

    _meta_lib_dont_use_me_use_register_meta_for_quantized = ...
    @register_meta(torch.ops.quantized.max_pool2d)
    def meta_quantized_max_pool2d(
        input, kernel_size, stride=..., padding=..., dilation=..., ceil_mode=...
    ) -> Tensor: ...
    @register_meta(torch.ops.quantized.int4mm_packed_weight_cpu)
    def meta_int4mm_packed_weight_cpu(x, w, q_group_size, q_scale_and_zeros): ...

def check_dim_size(tensor, dim, dim_size, size) -> None: ...
@register_meta(aten.avg_pool2d.default)
def meta_avg_pool2d(
    input,
    kernel_size,
    stride=...,
    padding=...,
    ceil_mode=...,
    count_include_pad=...,
    divisor_override=...,
) -> Tensor: ...
def avg_pool2d_backward_shape_check(
    input,
    gradOutput,
    nbatch,
    kH,
    kW,
    dH,
    dW,
    padH,
    padW,
    nInputPlane,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    mem_format,
) -> None: ...
@register_meta(aten.avg_pool2d_backward.default)
def meta_avg_pool2d_backward(
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
) -> Tensor: ...
@register_meta(aten.avg_pool3d)
@out_wrapper()
def meta_avg_pool3d(
    input,
    kernel_size,
    stride=...,
    padding=...,
    ceil_mode=...,
    count_include_pad=...,
    divisor_override=...,
): ...
@register_meta(aten.avg_pool3d_backward)
@out_wrapper("grad_input")
def meta_avg_pool3d_backward(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
): ...
@register_meta(aten._adaptive_avg_pool2d.default)
def meta_adaptive_avg_pool2d(self, output_size) -> Tensor: ...
@register_meta(aten._adaptive_avg_pool3d.default)
def meta_adaptive_avg_pool3d(self, output_size): ...
@register_meta(aten._adaptive_avg_pool2d_backward.default)
def meta__adaptive_avg_pool2d_backward(grad_out, self): ...
@register_meta(aten._adaptive_avg_pool3d_backward)
@out_wrapper("grad_input")
def meta__adaptive_avg_pool3d_backward(grad_output, self) -> Tensor: ...
@register_meta(aten.adaptive_max_pool2d)
@out_wrapper("out", "indices")
def meta_adaptive_max_pool2d(input, output_size) -> tuple[Any, Any]: ...
@register_meta(aten.adaptive_max_pool2d_backward)
@out_wrapper("grad_input")
def meta_adaptive_max_pool2d_backward(grad_output, input, indices): ...
@register_meta(aten.adaptive_max_pool3d)
@out_wrapper("out", "indices")
def meta_adaptive_max_pool3d(input, output_size) -> tuple[Any, Any]: ...
@register_meta(aten.adaptive_max_pool3d_backward)
@out_wrapper("grad_input")
def meta_adaptive_max_pool3d_backward(grad_output, input, indices): ...
@register_meta(aten.repeat_interleave.Tensor)
def meta_repeat_interleave_Tensor(repeats, output_size=...): ...
@register_meta([aten.complex.default, aten.complex.out])
@out_wrapper()
def meta_complex(real, imag) -> FakeTensor: ...
@register_meta([aten.nonzero_static.default, aten.nonzero_static.out])
@out_wrapper()
def nonzero_static(self, *, size, fill_value: int = ...): ...
@register_meta([torch.ops.aten.nonzero.default, torch.ops.aten.nonzero.out])
@out_wrapper()
def nonzero(self) -> Tensor: ...
@register_meta([aten.index.Tensor, aten._unsafe_index.Tensor])
def meta_index_Tensor(self, indices): ...
@register_meta([aten.convolution_backward.default])
def meta_convolution_backward(
    grad_output_,
    input_,
    weight_,
    bias_sizes_opt,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
) -> tuple[Any | None, Any | None, Any | None]: ...
@register_meta([aten.addbmm.default, aten.addbmm.out])
@out_wrapper(exact_dtype=True)
def meta_addbmm(self, batch1, batch2, *, beta=..., alpha=...): ...
@register_meta([aten.randint_like.Tensor])
def meta_randint_like(self, high, **kwargs): ...
@register_meta([aten._fused_adam_.default, aten._fused_adamw_.default])
def meta__fused_adam_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=...,
    found_inf=...,
) -> None: ...
@register_meta([aten._fused_adam.default])
def meta__fused_adam(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=...,
    found_inf=...,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor]]: ...
@register_meta([aten._int_mm])
@out_wrapper()
def meta__int_mm(a, b): ...
@register_meta([aten._convert_weight_to_int4pack])
def meta__convert_weight_to_int4pack(w, inner_k_tiles): ...
@register_meta([aten._convert_weight_to_int4pack_for_cpu])
def meta__convert_weight_to_int4pack_for_cpu(w, inner_k_tiles): ...
@register_meta([aten._weight_int4pack_mm])
def meta__weight_int4pack_mm(x, w, q_group_size, q_scale_and_zeros): ...
@register_meta([aten._weight_int4pack_mm_for_cpu])
def meta__weight_int4pack_mm_for_cpu(x, w, q_group_size, q_scale_and_zeros): ...
def kai_roundup(a: int, b: int) -> int: ...
def get_kai_packed_weight_size(n_bits, N, K, groupsize) -> None: ...
@register_meta([aten._dyn_quant_pack_4bit_weight])
def meta__dyn_quant_pack_4bit_weight(
    weights, scales_zeros, bias: Tensor | None, block_size, in_features, out_features
): ...
@register_meta([aten._dyn_quant_matmul_4bit])
def meta__dyn_quant_matmul_4bit(
    inp, packed_weights, block_size, in_features, out_features
): ...
@register_meta([aten._weight_int8pack_mm])
def meta__weight_int8pack_mm(x, w, q_scales): ...
@register_meta(aten._cdist_forward.default)
def meta_cdist_forward(x1, x2, p, compute_mode): ...
@register_meta(aten._cdist_backward)
@out_wrapper()
def meta_cdist_backward(grad, x1, x2, p, cdist) -> Tensor: ...
@register_meta(aten._embedding_bag.default)
def meta_embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=...,
    mode=...,
    sparse=...,
    per_sample_weights=...,
    include_last_offset=...,
    padding_idx=...,
) -> tuple[Any, Any, Any, Any]: ...
@register_meta(aten._embedding_bag_forward_only.default)
def meta_embedding_bag_forward_only(
    weight, indices, offsets, *args
) -> tuple[Any, Any, Any, Any]: ...
@register_meta([aten.nansum.default, aten.nansum.out])
@out_wrapper()
def meta_nansum(input, dims=..., keepdim=..., *, dtype=...): ...
@register_meta([aten.median.default, aten.nanmedian.default])
def meta_median(input): ...
@register_meta(
    [
        aten.median.dim,
        aten.median.dim_values,
        aten.nanmedian.dim,
        aten.nanmedian.dim_values,
        aten.mode.default,
        aten.mode.values,
    ]
)
@out_wrapper("values", "indices")
def meta_median_mode_dim(input, dim=..., keepdim=...) -> tuple[Any, Any]: ...
@register_meta(aten.logical_not_.default)
def meta_logical_not_(self): ...
@register_meta(aten.repeat.default)
def meta_repeat(self, repeats): ...
@register_meta(aten.zero_.default)
def meta_zero_(self): ...
@register_meta(
    [
        aten.mul_.Scalar,
        aten.div_.Scalar,
        aten.mul_.Tensor,
        aten.div_.Tensor,
        aten.logical_and_.default,
        aten.logical_or_.default,
        aten.logical_xor_.default,
    ]
)
def meta_binop_inplace(self, other): ...
@register_meta([aten.add_.Scalar, aten.sub_.Scalar, aten.add_.Tensor, aten.sub_.Tensor])
def meta_binop_inplace_alpha(self, other, alpha=...): ...
@register_meta([aten.add.Scalar, aten.sub.Scalar])
def meta_binop_alpha(self, other, alpha=...) -> FakeTensor: ...
@register_meta([aten.round.default, aten.round.decimals])
def meta_round(self, **kwargs) -> FakeTensor: ...
def shift_dtype_check(fn_name, self, val) -> None: ...
@register_meta([aten.__rshift__.Tensor, aten.__rshift__.Scalar])
def meta_rshifts(self, other) -> FakeTensor: ...
@register_meta([aten.__lshift__.Tensor, aten.__lshift__.Scalar])
def meta_lshifts(self, other) -> FakeTensor: ...
@register_meta(aten.zero.default)
def meta_zero(self): ...
@register_meta([aten.fill_.Tensor, aten.fill_.Scalar])
def meta_fill_(self, val): ...
@register_meta([aten.fill.Tensor, aten.fill.Scalar])
def meta_fill(self, val) -> Tensor: ...
@register_meta(aten.relu_.default)
def meta_relu_(self): ...
@register_meta(aten._add_relu.Tensor)
@out_wrapper()
def meta__add_relu(self, other, alpha=...) -> Tensor: ...
@register_meta([aten.rrelu_with_noise])
@out_wrapper()
def meta_rrelu_with_noise(
    self, noise, lower=..., upper=..., training=..., generator=...
) -> Tensor: ...
@register_meta([aten.rrelu_with_noise_functional])
def meta_rrelu_with_noise_functional(
    self, noise, lower=..., upper=..., training=..., generator=...
) -> tuple[Tensor, Tensor]: ...
@register_meta([aten.rrelu_with_noise_])
def meta_rrelu_with_noise_(self, lower=..., upper=..., training=..., generator=...): ...
@register_meta([aten.index_put.default, aten._unsafe_index_put.default])
def meta_index_put(self, indices, values, accumulate=...) -> Tensor: ...
@register_meta(aten.masked_fill_.Scalar)
def meta_masked_fill_(self, mask, value): ...
@register_meta(aten._masked_scale.default)
def meta__masked_scale(self, mask, scale): ...
@register_meta(aten.masked_scatter_)
def meta_masked_scatter_(self, mask, source): ...
@register_meta(aten.masked_scatter)
@out_wrapper()
def meta_masked_scatter(self, mask, source): ...
@register_meta(aten.masked_scatter_backward)
def meta_masked_scatter_backward(self, mask, sizes): ...
@register_meta(aten.index_put_.default)
def meta_index_put_(self, indices, values, accumulate=...): ...
def common_meta_baddbmm_bmm(
    batch1, batch2, is_bmm, self_baddbmm=..., out_dtype=...
): ...
@register_meta(aten.bmm.default)
def meta_bmm(self, mat2): ...
@register_meta(aten.bmm.dtype)
def meta_bmm_dtype(self, mat2, out_dtype): ...
def div_rtn(x, y): ...
def pooling_output_shape_pad_lr(
    inputSize, kernelSize, pad_l, pad_r, stride, dilation, ceil_mode
): ...
def pooling_output_shape(inputSize, kernelSize, pad, stride, dilation, ceil_mode): ...
def pool2d_shape_check(
    input,
    kH,
    kW,
    dH,
    dW,
    padH,
    padW,
    dilationH,
    dilationW,
    nInputPlane,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    memory_format,
) -> None: ...
def pool3d_shape_check(
    input: Tensor,
    nslices: int,
    kT: int,
    kH: int,
    kW: int,
    dT: int,
    dH: int,
    dW: int,
    pT: int,
    pH: int,
    pW: int,
    dilationT: int,
    dilationH: int,
    dilationW: int,
    itime: int,
    iheight: int,
    iwidth: int,
    otime: int,
    oheight: int,
    owidth: int,
    fn_name: str,
    check_input_size: bool = ...,
) -> None: ...
def max_pool3d_backward_shape_check(
    input,
    grad_output,
    indices,
    nslices,
    kT,
    kH,
    kW,
    dT,
    dH,
    dW,
    pT,
    pH,
    pW,
    dilationT,
    dilationH,
    dilationW,
    itime,
    iheight,
    iwidth,
    otime,
    oheight,
    owidth,
    fn_name,
) -> None: ...
def avg_pool3d_backward_shape_check(
    input: Tensor,
    grad_output: Tensor,
    nslices: int,
    kT: int,
    kH: int,
    kW: int,
    dT: int,
    dH: int,
    dW: int,
    pT: int,
    pH: int,
    pW: int,
    itime: int,
    iheight: int,
    iwidth: int,
    otime: int,
    oheight: int,
    owidth: int,
    fn_name: str,
) -> None: ...
def max_pool2d_checks_and_compute_shape(
    input, kernel_size, stride, padding, dilation, ceil_mode
) -> tuple[Any, Any, Any]: ...
@register_meta(aten.max_pool2d_with_indices_backward.default)
def meta_max_pool2d_with_indices_backward(
    grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
) -> Tensor: ...
@register_meta(aten.max_pool2d_with_indices.default)
def meta_max_pool2d_with_indices(
    input, kernel_size, stride=..., padding=..., dilation=..., ceil_mode=...
) -> tuple[Tensor, Tensor]: ...
@register_meta(aten.fractional_max_pool2d.default)
def meta_fractional_max_pool2d(
    self, kernel_size, output_size, random_samples
) -> tuple[Tensor, Tensor]: ...
@register_meta(aten.max_pool3d_with_indices)
@out_wrapper("out", "indices")
def meta_max_pool3d_with_indices(
    input, kernel_size, stride=..., padding=..., dilation=..., ceil_mode=...
) -> tuple[Any, Any]: ...
@register_meta(aten.max_pool3d_with_indices_backward)
@out_wrapper("grad_input")
def meta_max_pool3d_with_indices_backward(
    grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices
): ...
def check_grid_sampler_common(input: Tensor, grid: Tensor) -> None: ...

class GridSamplerInterpolation(Enum):
    BILINEAR = ...
    NEAREST = ...
    BICUBIC = ...

def check_grid_sampler_3d(
    input: Tensor, grid: Tensor, interpolation_mode: int
) -> None: ...
@register_meta(aten.grid_sampler_2d_backward.default)
def grid_sampler_2d_backward_meta(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
) -> tuple[Tensor | None, Tensor]: ...
@register_meta(aten.grid_sampler_3d)
@out_wrapper()
def grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners): ...
@register_meta(aten.grid_sampler_3d_backward)
@out_wrapper("grad_input", "grad_grid")
def grid_sampler_3d_backward(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
) -> tuple[Tensor | None, Tensor]: ...
@register_meta([aten.full.default])
def full(size, fill_value, *args, **kwargs) -> Tensor: ...
@register_meta(aten.zeros_like.default)
def zeros_like(
    self, dtype=..., layout=..., device=..., pin_memory=..., memory_format=...
) -> Tensor | Any: ...
@register_meta([aten.ones.default, aten.ones.out])
@out_wrapper()
def meta_ones(
    size, *, dtype=..., layout=..., device=..., pin_memory=..., requires_grad=...
) -> Tensor: ...
@register_meta([aten.zeros.default, aten.zeros.out])
@out_wrapper()
def meta_zeros(
    size, *, dtype=..., layout=..., device=..., pin_memory=..., requires_grad=...
) -> Tensor: ...
@register_meta(aten.select_scatter.default)
def meta_select_scatter(self, src, dim, index) -> Tensor: ...
@register_meta(aten.slice_scatter.default)
def meta_slice_scatter(self, src, dim=..., start=..., end=..., step=...) -> Tensor: ...
def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = ...) -> int: ...
def ensure_nonempty_size(t, dim) -> Literal[1]: ...
def gather_shape_check(self, dim, index) -> None: ...
@register_meta(aten.gather.default)
def meta_gather(self, dim, index, sparse_grad=...): ...
def get_operator_enum(
    reduce_, use_new_options=...
) -> (
    Literal[REDUCE_ADD, REDUCE_MULTIPLY, REDUCE_MEAN, REDUCE_MAXIMUM, REDUCE_MINIMUM]
    | None
): ...
def scatter_gather_dtype_check(method_name, self, index, src_opt=...) -> None: ...
def ensure_nonempty_dim(dim) -> int: ...
def scatter_shape_check(self, dim, index, src_opt=...) -> None: ...
def scatter_meta_impl(
    self, dim, index, src=..., reduce_=..., use_new_options=...
) -> None: ...
@register_meta(aten.scatter_add.default)
def meta_scatter_add(self, dim, index, src): ...
@register_meta(aten.scatter_add_)
def meta_scatter_add_(self, dim, index, src): ...
@register_meta(
    [
        aten.scatter.src,
        aten.scatter.value,
        aten.scatter.reduce,
        aten.scatter.value_reduce,
    ]
)
@out_wrapper()
def meta_scatter(self, dim, index, src_or_value, reduce=...): ...
@register_meta(
    [
        aten.scatter_.src,
        aten.scatter_.value,
        aten.scatter_.reduce,
        aten.scatter_.value_reduce,
    ]
)
def meta_scatter_(self, dim, index, src_or_value, reduce=...): ...
@register_meta([aten._scaled_dot_product_flash_attention])
def meta__scaled_dot_product_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = ...,
    is_causal: bool = ...,
    return_debug_mask: bool = ...,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor, None, None, int, int, Tensor, Tensor, Tensor]: ...
def alloc_with_matching_layout(query: Tensor, res_shape: tuple[int, ...]) -> Tensor: ...
@register_meta([aten._scaled_dot_product_cudnn_attention])
def meta__scaled_dot_product_cudnn_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Tensor | None,
    compute_log_sumexp: bool,
    dropout_p: float = ...,
    is_causal: bool = ...,
    return_debug_mask: bool = ...,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor, None, None, int, int, Tensor, Tensor, None]: ...
@register_meta([aten._scaled_dot_product_fused_attention_overrideable])
def meta__scaled_dot_product_fused_attention_overrideable(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Tensor | None = ...,
    dropout_p: float = ...,
    is_causal: bool = ...,
    return_debug_mask: bool = ...,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor, None, None, int, int, Tensor, Tensor, None]: ...
@register_meta([aten._scaled_dot_product_flash_attention_backward])
def meta__scaled_dot_product_flash_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: Tensor,
    philox_offset: Tensor,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_meta([aten._scaled_dot_product_flash_attention_for_cpu])
def meta__scaled_dot_product_flash_attention_for_cpu(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = ...,
    is_causal: bool = ...,
    attn_mask: Tensor | None = ...,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor]: ...
@register_meta([aten._scaled_dot_product_flash_attention_for_cpu_backward])
def meta__scaled_dot_product_flash_attention_for_cpu_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    dropout_p: float,
    is_causal: bool,
    attn_mask: Tensor | None = ...,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_meta([aten._scaled_dot_product_attention_math_for_mps])
def meta__scaled_dot_product_attention_math_for_mps(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = ...,
    dropout_p: float = ...,
    is_causal: bool = ...,
    dropout_mask: Tensor | None = ...,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor]: ...
@register_meta([aten._scaled_dot_product_efficient_attention])
def meta__scaled_dot_product_efficient_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Tensor | None,
    compute_log_sumexp: bool,
    dropout_p=...,
    is_causal: bool = ...,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
@register_meta([aten._scaled_dot_product_efficient_attention_backward])
def meta__scaled_dot_product_efficient_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Tensor | None,
    out: Tensor,
    logsumexp: Tensor,
    philox_seed: Tensor,
    philox_offset: Tensor,
    dropout_p: float,
    grad_input_mask: list[bool],
    is_causal: bool = ...,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]: ...
@register_meta([aten._scaled_dot_product_cudnn_attention_backward])
def meta__scaled_dot_product_cudnn_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    philox_seed: Tensor,
    philox_offset: Tensor,
    attn_bias: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_meta([aten._flash_attention_forward])
def meta__flash_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cum_seq_q: Tensor | None,
    cum_seq_k: Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    scale: float | None = ...,
    window_size_left: int | None = ...,
    window_size_right: int | None = ...,
    seqused_k: Tensor | None = ...,
    alibi_slopes: Tensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
@register_meta([aten._flash_attention_backward])
def meta__flash_attention_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: Tensor,
    philox_offset: Tensor,
    scale: float | None = ...,
    window_size_left: int | None = ...,
    window_size_right: int | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_meta([aten._efficient_attention_forward])
def meta__efficient_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Tensor | None,
    cu_seqlens_q: Tensor | None,
    cu_seqlens_k: Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    dropout_p: float,
    custom_mask_type: int,
    compute_log_sumexp: bool = ...,
    scale: float | None = ...,
    causal_diagonal: Tensor | None = ...,
    seqlen_k: Tensor | None = ...,
    window_size: int | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int, int]: ...
@register_meta([aten._efficient_attention_backward])
def meta__efficient_attention_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Tensor | None,
    cu_seqlens_q: Tensor | None,
    cu_seqlens_k: Tensor | None,
    max_seqlen_q: torch.SymInt,
    max_seqlen_k: torch.SymInt,
    logsumexp: Tensor,
    dropout_p: float,
    philox_seed: Tensor,
    philox_offset: Tensor,
    custom_mask_type: int,
    bias_requires_grad: bool,
    scale: float | None = ...,
    num_splits_key: int | None = ...,
    shared_storage_dqdkdv: bool = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
@register_meta([aten._scaled_mm.default])
def meta_scaled_mm(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = ...,
    scale_result: torch.Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
    use_fast_accum: bool = ...,
) -> Tensor: ...
@register_meta([aten.scatter_reduce.two, aten.scatter_reduce.two_out])
@out_wrapper()
def meta_scatter_reduce_two(self, dim, index, src, reduce, include_self=...): ...
@register_meta(aten.scatter_reduce_.two)
def meta_scatter_reduce__two(self, dim, index, src, reduce, include_self=...): ...
@register_meta([aten.multinomial.default, aten.multinomial.out])
@out_wrapper()
def meta_multinomial(
    input, num_samples, replacement=..., *, generator=...
) -> Tensor: ...
def multiply_integers(vs) -> Literal[1]: ...
def upsample_common_check(
    input_size, output_size, num_spatial_dims
) -> tuple[Any, Any, *tuple[Any, ...]]: ...
@register_meta(
    [aten.upsample_nearest1d.default, aten._upsample_nearest_exact1d.default]
)
def upsample_nearest1d(input, output_size, scales=...): ...
@register_meta(
    [aten.upsample_nearest2d.default, aten._upsample_nearest_exact2d.default]
)
def upsample_nearest2d(input, output_size, scales_h=..., scales_w=...): ...
@register_meta(
    [
        aten.upsample_nearest2d_backward.default,
        aten._upsample_nearest_exact2d_backward.default,
    ]
)
def upsample_nearest2d_backward(
    grad_output: Tensor,
    output_size: Sequence[int | torch.SymInt],
    input_size: Sequence[int | torch.SymInt],
    scales_h: float | None = ...,
    scales_w: float | None = ...,
) -> Tensor: ...
@register_meta(
    [aten.upsample_nearest3d.default, aten._upsample_nearest_exact3d.default]
)
def upsample_nearest3d(
    input, output_size, scales_d=..., scales_h=..., scales_w=...
): ...
@register_meta(
    [aten.sort.default, aten.sort.stable, aten.sort.values, aten.sort.values_stable]
)
def meta_sort(
    self, stable=..., dim=..., descending=..., values=..., indices=...
) -> tuple[Tensor, Tensor]: ...
def rnn_cell_checkSizes(
    input_gates, hidden_gates, input_bias, hidden_bias, factor, prev_hidden
) -> None: ...
@register_meta(aten.mkldnn_rnn_layer.default)
def mkldnn_rnn_layer(
    input,
    w0,
    w1,
    w2,
    w3,
    hx_,
    cx_,
    reverse,
    batch_sizes,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    bidirectional,
    batch_first,
    train,
) -> tuple[Any, Tensor | Any, Tensor | Any, Tensor]: ...
def zero_numel_check_dims(self, dim, fn_name) -> None: ...
def check_argmax_argmin(name, self, dim) -> None: ...
@register_meta([aten.argmax.default, aten.argmin.default])
def argmax_argmin_meta(self, dim=..., keepdim=...): ...
@register_meta(aten.scalar_tensor.default)
def scalar_tensor(s, dtype=..., layout=..., device=..., pin_memory=...) -> Tensor: ...
@register_meta(aten.topk.default)
def topk_meta(self, k, dim=..., largest=..., sorted=...) -> tuple[Any, Any]: ...
@register_meta(aten._segment_reduce_backward)
@out_wrapper()
def meta__segment_reduce_backward(
    grad, output, data, reduce, lengths=..., offsets=..., axis=..., initial=...
) -> Tensor: ...
@register_meta([aten.kthvalue.default, aten.kthvalue.values])
@out_wrapper("values", "indices")
def kthvalue_meta(self, k, dim=..., keepdim=...) -> tuple[Any, Any]: ...

legacy_contiguous_memory_format = ...

def checkLSTMBackwardSizes(grad_hy, grad_cy, cx, cy, workspace) -> None: ...
@register_meta(aten.linear_backward.default)
def linear_backward(
    input_, grad_output_, weight_, output_mask
) -> tuple[Any | None, Any | None, Any | None]: ...
@register_meta(aten.pixel_shuffle.default)
def meta_pixel_shuffle(self, upscale_factor): ...
@register_meta(aten.mkldnn_rnn_layer_backward.default)
def mkldnn_rnn_layer_backward(
    input,
    weight0,
    weight1,
    weight2,
    weight3,
    hx_,
    cx_tmp,
    output,
    hy_,
    cy_,
    grad_output_r_opt,
    grad_hy_r_opt,
    grad_cy_r_opt,
    reverse,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    train,
    bidirectional,
    batch_sizes,
    batch_first,
    workspace,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]: ...
@register_meta([aten.bucketize.Tensor, aten.bucketize.Tensor_out])
@out_wrapper()
def meta_bucketize(self, boundaries, *, out_int32=..., right=...) -> Tensor: ...
@register_meta([aten.histc])
@out_wrapper()
def meta_histc(input, bins=..., min=..., max=...) -> Tensor: ...
@register_meta(
    [aten._upsample_bilinear2d_aa.default, aten._upsample_bicubic2d_aa.default]
)
def meta_upsample_bimode2d_aa(
    input, output_size, align_corners, scales_h=..., scales_w=...
): ...
@register_meta([aten._upsample_bilinear2d_aa_backward.default])
def meta_upsample_bimode2d_aa_backward(
    grad_output, output_size, input_size, align_corners, scales_h=..., scales_w=...
): ...
@register_meta([aten.nan_to_num.default, aten.nan_to_num.out])
@out_wrapper()
def nan_to_num(self, nan=..., posinf=..., neginf=...) -> Tensor: ...
@register_meta(torch.ops.aten.transpose_)
def transpose_(self, dim0, dim1): ...
@register_meta(torch.ops.aten.t_)
def t_(self): ...
@register_meta(aten.searchsorted)
@out_wrapper()
def meta_searchsorted(
    sorted_sequence, self, *, out_int32=..., right=..., side=..., sorter=...
) -> Tensor: ...
@register_meta(aten.embedding_dense_backward)
def meta_embedding_dense_backward(
    grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
): ...
@register_meta(aten._embedding_bag_backward)
def meta_embedding_bag_backward(
    grad,
    indices,
    offsets,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    padding_idx=...,
) -> Any: ...
@register_meta(aten._embedding_bag_dense_backward)
def meta_embedding_bag_dense_backward(
    grad,
    indices,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    per_sample_weights,
    padding_idx=...,
): ...
@register_meta(aten._embedding_bag_per_sample_weights_backward)
def meta_embedding_bag_per_sample_weights_backward(
    grad, weight, indices, offsets, offset2bag, mode, padding_idx=...
): ...
@register_meta(aten.isin)
@out_wrapper()
def meta_isin(elements, test_elements, *, assume_unique=..., invert=...) -> Tensor: ...
@register_meta(aten.polygamma)
@out_wrapper()
def meta_polygamma(n: int, self: Tensor) -> Tensor: ...
@register_meta(aten._local_scalar_dense)
def meta_local_scalar_dense(self: Tensor): ...
@register_meta(aten.silu)
@out_wrapper(exact_dtype=True)
def silu(self: Tensor) -> Tensor: ...
@register_meta(aten.sigmoid)
@out_wrapper()
def sigmoid(self: Tensor) -> Tensor: ...
@register_meta(aten._grouped_mm)
@out_wrapper()
def meta_grouped_mm(
    mat_a: Tensor,
    mat_b: Tensor,
    offs: Tensor | None = ...,
    bias: Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
) -> Tensor: ...
@register_meta([aten._scaled_grouped_mm])
def meta_scaled_grouped_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: torch.Tensor | None = ...,
    bias: torch.Tensor | None = ...,
    scale_result: torch.Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
    use_fast_accum: bool = ...,
) -> Tensor: ...
@register_meta(aten._softmax)
@out_wrapper()
def softmax(x: Tensor, dim: int, half_to_float: bool) -> Tensor: ...
@register_meta(aten.embedding)
@out_wrapper()
def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: int = ...,
    scale_grad_by_freq: bool = ...,
    sparse: bool = ...,
) -> Tensor: ...
@register_meta(aten._jagged_to_padded_dense_forward.default)
def meta__jagged_to_padded_dense_forward(
    values: Tensor,
    offsets: list[Tensor],
    max_lengths: list[int],
    padding_value: float = ...,
) -> Tensor: ...
@register_meta(aten.lerp)
@out_wrapper()
def lerp(start, end, weight) -> FakeTensor: ...
@register_meta(aten.addcmul)
@out_wrapper()
def addcmul(input, tensor1, tensor2, *, value=...) -> FakeTensor: ...
@register_meta(aten.addcdiv)
@out_wrapper()
def addcdiv(input, tensor1, tensor2, *, value=...) -> FakeTensor: ...

lerp_ = ...
addcmul_ = ...
addcdiv_ = ...

def activate_meta() -> None: ...
