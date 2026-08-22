from collections.abc import Sequence
from types import EllipsisType
from typing import Any, Literal, overload

from torch import Generator, SymInt, Tensor, memory_format
from torch._C import _dtype, _int
from torch._prims_common import DeviceLikeType
from torch.types import Number, _bool, _complex, _dtype, _float, _int, _layout, _size

import torch

__all__ = [
    "__and__",
    "__lshift__",
    "__or__",
    "__rshift__",
    "__xor__",
    "_adaptive_avg_pool2d",
    "_adaptive_avg_pool3d",
    "_add_batch_dim",
    "_add_relu",
    "_add_relu_",
    "_addmm_activation",
    "_aminmax",
    "_amp_foreach_non_finite_check_and_unscale_",
    "_amp_update_scale_",
    "_assert_async",
    "_assert_scalar",
    "_assert_tensor_metadata",
    "_batch_norm_impl_index",
    "_cast_Byte",
    "_cast_Char",
    "_cast_Double",
    "_cast_Float",
    "_cast_Half",
    "_cast_Int",
    "_cast_Long",
    "_cast_Short",
    "_choose_qparams_per_tensor",
    "_chunk_cat",
    "_coalesce",
    "_compute_linear_combination",
    "_conj",
    "_conj_copy",
    "_conj_physical",
    "_convert_indices_from_coo_to_csr",
    "_convert_indices_from_csr_to_coo",
    "_convert_weight_to_int4pack",
    "_convert_weight_to_int4pack_for_cpu",
    "_convolution",
    "_convolution_mode",
    "_copy_from",
    "_copy_from_and_resize",
    "_cslt_compress",
    "_cslt_sparse_mm",
    "_cslt_sparse_mm_search",
    "_ctc_loss",
    "_cudnn_ctc_loss",
    "_cudnn_init_dropout_state",
    "_cudnn_rnn",
    "_cudnn_rnn_flatten_weight",
    "_cufft_clear_plan_cache",
    "_cufft_get_plan_cache_max_size",
    "_cufft_get_plan_cache_size",
    "_cufft_set_plan_cache_max_size",
    "_cummax_helper",
    "_cummin_helper",
    "_debug_has_internal_overlap",
    "_dim_arange",
    "_dirichlet_grad",
    "_disable_functionalization",
    "_dyn_quant_matmul_4bit",
    "_dyn_quant_pack_4bit_weight",
    "_efficientzerotensor",
    "_embedding_bag",
    "_embedding_bag_forward_only",
    "_empty_affine_quantized",
    "_empty_per_channel_affine_quantized",
    "_enable_functionalization",
    "_euclidean_dist",
    "_fake_quantize_learnable_per_channel_affine",
    "_fake_quantize_learnable_per_tensor_affine",
    ...,
    "_fft_c2c",
    "_fft_c2r",
    "_fft_r2c",
    "_fill_mem_eff_dropout_mask_",
    "_foobar",
    "_foreach_abs",
    "_foreach_abs_",
    "_foreach_acos",
    "_foreach_acos_",
    "_foreach_add",
    "_foreach_add_",
    "_foreach_addcdiv",
    "_foreach_addcdiv_",
    "_foreach_addcmul",
    "_foreach_addcmul_",
    "_foreach_asin",
    "_foreach_asin_",
    "_foreach_atan",
    "_foreach_atan_",
    "_foreach_ceil",
    "_foreach_ceil_",
    "_foreach_clamp_max",
    "_foreach_clamp_max_",
    "_foreach_clamp_min",
    "_foreach_clamp_min_",
    "_foreach_copy_",
    "_foreach_cos",
    "_foreach_cos_",
    "_foreach_cosh",
    "_foreach_cosh_",
    "_foreach_div",
    "_foreach_div_",
    "_foreach_erf",
    "_foreach_erf_",
    "_foreach_erfc",
    "_foreach_erfc_",
    "_foreach_exp",
    "_foreach_exp_",
    "_foreach_expm1",
    "_foreach_expm1_",
    "_foreach_floor",
    "_foreach_floor_",
    "_foreach_frac",
    "_foreach_frac_",
    "_foreach_lerp",
    "_foreach_lerp_",
    "_foreach_lgamma",
    "_foreach_lgamma_",
    "_foreach_log",
    "_foreach_log1p",
    "_foreach_log1p_",
    "_foreach_log2",
    "_foreach_log2_",
    "_foreach_log10",
    "_foreach_log10_",
    "_foreach_log_",
    "_foreach_max",
    "_foreach_maximum",
    "_foreach_maximum_",
    "_foreach_minimum",
    "_foreach_minimum_",
    "_foreach_mul",
    "_foreach_mul_",
    "_foreach_neg",
    "_foreach_neg_",
    "_foreach_norm",
    "_foreach_pow",
    "_foreach_pow_",
    "_foreach_reciprocal",
    "_foreach_reciprocal_",
    "_foreach_round",
    "_foreach_round_",
    "_foreach_rsqrt",
    "_foreach_rsqrt_",
    "_foreach_sigmoid",
    "_foreach_sigmoid_",
    "_foreach_sign",
    "_foreach_sign_",
    "_foreach_sin",
    "_foreach_sin_",
    "_foreach_sinh",
    "_foreach_sinh_",
    "_foreach_sqrt",
    "_foreach_sqrt_",
    "_foreach_sub",
    "_foreach_sub_",
    "_foreach_tan",
    "_foreach_tan_",
    "_foreach_tanh",
    "_foreach_tanh_",
    "_foreach_trunc",
    "_foreach_trunc_",
    "_foreach_zero_",
    "_from_functional_tensor",
    "_functional_assert_async",
    "_functional_assert_scalar",
    "_functional_sym_constrain_range",
    "_functional_sym_constrain_range_for_size",
    "_functionalize_apply_view_metas",
    ...,
    ...,
    "_functionalize_commit_update",
    "_functionalize_has_metadata_mutation",
    "_functionalize_inductor_storage_resized_counter",
    "_functionalize_is_symbolic",
    "_functionalize_mark_mutation_hidden_from_autograd",
    "_functionalize_mark_storage_changed",
    "_functionalize_mutation_counter",
    "_functionalize_replace",
    "_functionalize_storage_changed_counter",
    "_functionalize_sync",
    "_functionalize_unsafe_set",
    "_functionalize_was_inductor_storage_resized",
    "_functionalize_was_storage_changed",
    "_fused_adagrad_",
    "_fused_adam_",
    "_fused_adamw_",
    "_fused_dropout",
    "_fused_moving_avg_obs_fq_helper",
    "_fused_rms_norm",
    "_fused_sdp_choice",
    "_fused_sgd_",
    "_fw_primal_copy",
    "_grid_sampler_2d_cpu_fallback",
    "_grouped_mm",
    "_has_compatible_shallow_copy_type",
    "_histogramdd_bin_edges",
    "_histogramdd_from_bin_cts",
    "_histogramdd_from_bin_tensors",
    "_index_put_impl_",
    "_indices_copy",
    "_int_mm",
    "_is_all_true",
    "_is_any_true",
    "_is_functional_tensor",
    "_is_functional_tensor_base",
    "_is_zerotensor",
    "_lazy_clone",
    "_linalg_check_errors",
    "_linalg_det",
    "_linalg_eigh",
    "_linalg_slogdet",
    "_linalg_solve_ex",
    "_linalg_svd",
    "_log_softmax",
    "_log_softmax_backward_data",
    "_logcumsumexp",
    "_lstm_mps",
    "_lu_with_info",
    "_make_dep_token",
    "_make_dual",
    "_make_dual_copy",
    "_make_per_channel_quantized_tensor",
    "_make_per_tensor_quantized_tensor",
    "_masked_scale",
    "_masked_softmax",
    "_mixed_dtypes_linear",
    "_mkldnn_reshape",
    "_mkldnn_transpose",
    "_mkldnn_transpose_",
    "_mps_convolution",
    "_mps_convolution_transpose",
    "_native_batch_norm_legit",
    "_native_batch_norm_legit_no_training",
    "_native_multi_head_attention",
    "_neg_view",
    "_neg_view_copy",
    "_nested_compute_contiguous_strides_offsets",
    "_nested_from_padded",
    "_nested_from_padded_and_nested_example",
    "_nested_from_padded_tensor",
    "_nested_get_jagged_dummy",
    "_nested_get_lengths",
    "_nested_get_max_seqlen",
    "_nested_get_min_seqlen",
    "_nested_get_offsets",
    "_nested_get_ragged_idx",
    "_nested_get_values",
    "_nested_get_values_copy",
    "_nested_tensor_from_mask",
    "_nested_tensor_from_mask_left_aligned",
    "_nested_tensor_from_tensor_list",
    "_nested_tensor_softmax_with_shape",
    "_nested_view_from_buffer",
    "_nested_view_from_buffer_copy",
    "_nested_view_from_jagged",
    "_nested_view_from_jagged_copy",
    "_nnpack_available",
    "_nnpack_spatial_convolution",
    "_pack_padded_sequence",
    "_pad_packed_sequence",
    "_pin_memory",
    "_prelu_kernel",
    "_print",
    "_propagate_xla_data",
    "_remove_batch_dim",
    "_reshape_alias_copy",
    "_reshape_from_tensor",
    "_resize_output_",
    "_rowwise_prune",
    "_safe_softmax",
    "_sample_dirichlet",
    "_saturate_weight_to_fp16",
    "_scaled_dot_product_attention_math",
    "_scaled_dot_product_attention_math_for_mps",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_grouped_mm",
    "_scaled_mm",
    "_shape_as_tensor",
    "_sobol_engine_draw",
    "_sobol_engine_ff_",
    "_sobol_engine_initialize_state_",
    "_sobol_engine_scramble_",
    "_softmax",
    "_softmax_backward_data",
    "_sparse_broadcast_to",
    "_sparse_broadcast_to_copy",
    "_sparse_csr_prod",
    "_sparse_csr_sum",
    "_sparse_log_softmax_backward_data",
    "_sparse_semi_structured_addmm",
    "_sparse_semi_structured_apply",
    "_sparse_semi_structured_apply_dense",
    "_sparse_semi_structured_linear",
    "_sparse_semi_structured_mm",
    "_sparse_semi_structured_tile",
    "_sparse_softmax_backward_data",
    "_sparse_sparse_matmul",
    "_sparse_sum",
    "_stack",
    "_standard_gamma",
    "_standard_gamma_grad",
    "_sync",
    "_test_autograd_multiple_dispatch",
    "_test_autograd_multiple_dispatch_view",
    "_test_autograd_multiple_dispatch_view_copy",
    "_test_check_tensor",
    "_test_functorch_fallback",
    "_test_parallel_materialize",
    "_test_serialization_subcmul",
    "_to_cpu",
    "_to_functional_tensor",
    "_to_sparse_semi_structured",
    "_transform_bias_rescale_qkv",
    "_transformer_encoder_layer_fwd",
    "_trilinear",
    "_triton_multi_head_attention",
    "_triton_scaled_dot_attention",
    "_unique",
    "_unique2",
    "_unpack_dual",
    "_unsafe_index",
    "_unsafe_index_put",
    "_unsafe_masked_index",
    "_unsafe_masked_index_put_accumulate",
    "_use_cudnn_ctc_loss",
    "_use_cudnn_rnn_flatten_weight",
    "_validate_compressed_sparse_indices",
    "_validate_sparse_bsc_tensor_args",
    "_validate_sparse_bsr_tensor_args",
    "_validate_sparse_compressed_tensor_args",
    "_validate_sparse_coo_tensor_args",
    "_validate_sparse_csc_tensor_args",
    "_validate_sparse_csr_tensor_args",
    "_values_copy",
    "_weight_int4pack_mm",
    "_weight_int4pack_mm_for_cpu",
    "_weight_int4pack_mm_with_scales_and_zeros",
    "_weight_int8pack_mm",
    "_weight_norm",
    "_weight_norm_interface",
    "_wrapped_linear_prepack",
    "_wrapped_quantized_linear_prepacked",
    "abs",
    "abs_",
    "absolute",
    "acos",
    "acos_",
    "acosh",
    "acosh_",
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "add",
    "addbmm",
    "addcdiv",
    "addcmul",
    "addmm",
    "addmv",
    "addmv_",
    "addr",
    "adjoint",
    "affine_grid_generator",
    "alias_copy",
    "all",
    "allclose",
    "alpha_dropout",
    "alpha_dropout_",
    "amax",
    "amin",
    "aminmax",
    "angle",
    "any",
    "arange",
    "arccos",
    "arccos_",
    "arccosh",
    "arccosh_",
    "arcsin",
    "arcsin_",
    "arcsinh",
    "arcsinh_",
    "arctan",
    "arctan2",
    "arctan_",
    "arctanh",
    "arctanh_",
    "argmax",
    "argmin",
    "argsort",
    "argwhere",
    "as_strided",
    "as_strided_",
    "as_strided_copy",
    "as_strided_scatter",
    "as_tensor",
    "asarray",
    "asin",
    "asin_",
    "asinh",
    "asinh_",
    "atan",
    "atan2",
    "atan_",
    "atanh",
    "atanh_",
    "avg_pool1d",
    "baddbmm",
    "bartlett_window",
    "batch_norm",
    "batch_norm_backward_elemt",
    "batch_norm_backward_reduce",
    "batch_norm_elemt",
    "batch_norm_gather_stats",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_stats",
    "batch_norm_update_stats",
    "bernoulli",
    "bilinear",
    "binary_cross_entropy_with_logits",
    "bincount",
    "binomial",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "blackman_window",
    "bmm",
    "broadcast_to",
    "bucketize",
    "can_cast",
    "cat",
    "ccol_indices_copy",
    "ceil",
    "ceil_",
    "celu",
    "celu_",
    "channel_shuffle",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "choose_qparams_optimized",
    "chunk",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clip",
    "clip_",
    "clone",
    "col_indices_copy",
    "column_stack",
    "combinations",
    "complex",
    "concat",
    "concatenate",
    "conj",
    "conj_physical",
    "conj_physical_",
    "constant_pad_nd",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convolution",
    "copysign",
    "corrcoef",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosine_embedding_loss",
    "cosine_similarity",
    "count_nonzero",
    "cov",
    "cross",
    "crow_indices_copy",
    "ctc_loss",
    "cudnn_affine_grid_generator",
    "cudnn_batch_norm",
    "cudnn_convolution",
    "cudnn_convolution_add_relu",
    "cudnn_convolution_relu",
    "cudnn_convolution_transpose",
    "cudnn_grid_sampler",
    "cudnn_is_acceptable",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "cumulative_trapezoid",
    "deg2rad",
    "deg2rad_",
    "dequantize",
    "det",
    "detach",
    "detach_",
    "detach_copy",
    "diag",
    "diag_embed",
    "diagflat",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "diff",
    "digamma",
    "dist",
    "div",
    "divide",
    "dot",
    "dropout",
    "dropout_",
    "dsmm",
    "dsplit",
    "dstack",
    "embedding",
    "embedding_bag",
    "embedding_renorm_",
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_quantized",
    "empty_strided",
    "eq",
    "equal",
    "erf",
    "erf_",
    "erfc",
    "erfc_",
    "erfinv",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "expand_copy",
    "expm1",
    "expm1_",
    "eye",
    "fake_quantize_per_channel_affine",
    "fake_quantize_per_tensor_affine",
    "fbgemm_linear_fp16_weight",
    "fbgemm_linear_fp16_weight_fp32_activation",
    "fbgemm_linear_int8_weight",
    "fbgemm_linear_int8_weight_fp32_activation",
    "fbgemm_linear_quantize_weight",
    "fbgemm_pack_gemm_matrix_fp16",
    "fbgemm_pack_quantized_matrix",
    "feature_alpha_dropout",
    "feature_alpha_dropout_",
    "feature_dropout",
    "feature_dropout_",
    "fill",
    "fill_",
    "fix",
    "fix_",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "float_power",
    "floor",
    "floor_",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frac",
    "frac_",
    "frexp",
    "frobenius_norm",
    "from_file",
    "from_numpy",
    "frombuffer",
    "full",
    "full_like",
    "fused_moving_avg_obs_fake_quant",
    "gather",
    "gcd",
    "gcd_",
    "ge",
    "geqrf",
    "ger",
    "get_default_dtype",
    "get_num_interop_threads",
    "get_num_threads",
    "gradient",
    "greater",
    "greater_equal",
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
    "group_norm",
    "gru",
    "gru_cell",
    "gt",
    "hamming_window",
    "hann_window",
    "hardshrink",
    "hash_tensor",
    "heaviside",
    "hinge_embedding_loss",
    "histc",
    "histogram",
    "histogramdd",
    "hsmm",
    "hsplit",
    "hspmm",
    "hstack",
    "hypot",
    "i0",
    "i0_",
    "igamma",
    "igammac",
    "imag",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_put_",
    "index_reduce",
    "index_select",
    "indices_copy",
    "init_num_threads",
    "inner",
    "instance_norm",
    "int_repr",
    "inverse",
    "is_complex",
    "is_conj",
    "is_distributed",
    "is_floating_point",
    "is_grad_enabled",
    "is_inference",
    "is_inference_mode_enabled",
    "is_neg",
    "is_nonzero",
    "is_same_size",
    "is_signed",
    "is_vulkan_available",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "istft",
    "kaiser_window",
    "kl_div",
    "kron",
    "kthvalue",
    "layer_norm",
    "lcm",
    "lcm_",
    "ldexp",
    "ldexp_",
    "le",
    "lerp",
    "less",
    "less_equal",
    "lgamma",
    "linspace",
    "log",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log10",
    "log10_",
    "log_",
    "log_softmax",
    "logaddexp",
    "logaddexp2",
    "logcumsumexp",
    "logdet",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "logit_",
    "logspace",
    "logsumexp",
    "lstm",
    "lstm_cell",
    "lt",
    "lu_solve",
    "lu_unpack",
    "margin_ranking_loss",
    "masked_fill",
    "masked_scatter",
    "masked_select",
    "matmul",
    "matrix_exp",
    "matrix_power",
    "max",
    "max_pool1d",
    "max_pool1d_with_indices",
    "max_pool2d",
    "max_pool3d",
    "maximum",
    "mean",
    "median",
    "min",
    "minimum",
    "miopen_batch_norm",
    "miopen_convolution",
    "miopen_convolution_add_relu",
    "miopen_convolution_relu",
    "miopen_convolution_transpose",
    "miopen_depthwise_convolution",
    "miopen_rnn",
    "mkldnn_adaptive_avg_pool2d",
    "mkldnn_convolution",
    "mkldnn_linear_backward_weights",
    "mkldnn_max_pool2d",
    "mkldnn_max_pool3d",
    "mkldnn_rnn_layer",
    "mm",
    "mode",
    "moveaxis",
    "movedim",
    "msort",
    "mul",
    "multinomial",
    "multiply",
    "mv",
    "mvlgamma",
    "nan_to_num",
    "nan_to_num_",
    "nanmean",
    "nanmedian",
    "nanquantile",
    "nansum",
    "narrow",
    "narrow_copy",
    "native_batch_norm",
    "native_channel_shuffle",
    "native_dropout",
    "native_group_norm",
    "native_layer_norm",
    "native_norm",
    "ne",
    "neg",
    "neg_",
    "negative",
    "negative_",
    "nextafter",
    "nonzero",
    "nonzero_static",
    "norm_except_dim",
    "normal",
    "not_equal",
    "nuclear_norm",
    "numel",
    "ones",
    "ones_like",
    "orgqr",
    "ormqr",
    "outer",
    "pairwise_distance",
    "pdist",
    "permute",
    "permute_copy",
    "pinverse",
    "pixel_shuffle",
    "pixel_unshuffle",
    "poisson",
    "poisson_nll_loss",
    "polar",
    "polygamma",
    "positive",
    "pow",
    "prelu",
    "prod",
    "promote_types",
    "put",
    "q_per_channel_axis",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "q_scale",
    "q_zero_point",
    "qr",
    "quantile",
    "quantize_per_channel",
    "quantize_per_tensor",
    "quantize_per_tensor_dynamic",
    "quantized_batch_norm",
    "quantized_gru_cell",
    "quantized_lstm_cell",
    "quantized_max_pool1d",
    "quantized_max_pool2d",
    "quantized_max_pool3d",
    "quantized_rnn_relu_cell",
    "quantized_rnn_tanh_cell",
    "rad2deg",
    "rad2deg_",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "range",
    "ravel",
    "real",
    "reciprocal",
    "reciprocal_",
    "relu",
    "relu_",
    "remainder",
    "renorm",
    "repeat_interleave",
    "reshape",
    "resize_as_",
    "resize_as_sparse_",
    "resolve_conj",
    "resolve_neg",
    "result_type",
    "rms_norm",
    "rnn_relu",
    "rnn_relu_cell",
    "rnn_tanh",
    "rnn_tanh_cell",
    "roll",
    "rot90",
    "round",
    "round_",
    "row_indices_copy",
    "row_stack",
    "rrelu",
    "rrelu_",
    "rsqrt",
    "rsqrt_",
    "rsub",
    "saddmm",
    "scalar_tensor",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "searchsorted",
    "segment_reduce",
    "select",
    "select_copy",
    "select_scatter",
    "selu",
    "selu_",
    "set_flush_denormal",
    "set_num_interop_threads",
    "set_num_threads",
    "sgn",
    "sigmoid",
    "sigmoid_",
    "sign",
    "signbit",
    "sin",
    "sin_",
    "sinc",
    "sinc_",
    "sinh",
    "sinh_",
    "slice_copy",
    "slice_inverse",
    "slice_scatter",
    "slogdet",
    "smm",
    "softmax",
    "sort",
    "sparse_bsc_tensor",
    "sparse_bsr_tensor",
    "sparse_compressed_tensor",
    "sparse_coo_tensor",
    "sparse_csc_tensor",
    "sparse_csr_tensor",
    "split_copy",
    "split_with_sizes",
    "split_with_sizes_copy",
    "spmm",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "squeeze",
    "squeeze_copy",
    "sspaddmm",
    "stack",
    "std",
    "std_mean",
    "sub",
    "subtract",
    "sum",
    "svd",
    "swapaxes",
    "swapdims",
    "sym_constrain_range",
    "sym_constrain_range_for_size",
    "t",
    "t_copy",
    "take",
    "take_along_dim",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tensor",
    "tensor_split",
    "threshold",
    "threshold_",
    "tile",
    "topk",
    "trace",
    "transpose",
    "transpose_copy",
    "trapezoid",
    "trapz",
    "triangular_solve",
    "tril",
    "tril_indices",
    "triplet_margin_loss",
    "triu",
    "triu_indices",
    "true_divide",
    "trunc",
    "trunc_",
    "unbind",
    "unbind_copy",
    "unflatten",
    "unfold_copy",
    "unique_dim",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    "unsqueeze",
    "unsqueeze_copy",
    "values_copy",
    "vander",
    "var",
    "var_mean",
    "vdot",
    "view_as_complex",
    "view_as_complex_copy",
    "view_as_real",
    "view_as_real_copy",
    "view_copy",
    "vsplit",
    "vstack",
    "where",
    "xlogy",
    "xlogy_",
    "zero_",
    "zeros",
    "zeros_like",
]

@overload
def __and__(input: Tensor, other: Tensor) -> Tensor: ...
@overload
def __and__(input: Tensor, other: Number | _complex) -> Tensor: ...
@overload
def __lshift__(input: Tensor, other: Tensor) -> Tensor: ...
@overload
def __lshift__(input: Tensor, other: Number | _complex) -> Tensor: ...
@overload
def __or__(input: Tensor, other: Tensor) -> Tensor: ...
@overload
def __or__(input: Tensor, other: Number | _complex) -> Tensor: ...
@overload
def __rshift__(input: Tensor, other: Tensor) -> Tensor: ...
@overload
def __rshift__(input: Tensor, other: Number | _complex) -> Tensor: ...
@overload
def __xor__(input: Tensor, other: Tensor) -> Tensor: ...
@overload
def __xor__(input: Tensor, other: Number | _complex) -> Tensor: ...
def _foreach_copy_(
    self: Sequence[Tensor], src: Sequence[Tensor], non_blocking: _bool = ...
) -> None: ...
def abs(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def abs_(input: Tensor) -> Tensor: ...
def absolute(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def acos(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def acos_(input: Tensor) -> Tensor: ...
def acosh(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def acosh_(input: Tensor) -> Tensor: ...
def adaptive_avg_pool1d(input: Tensor, output_size: _int | _size) -> Tensor: ...
def adaptive_max_pool1d(
    input: Tensor, output_size: _int | _size
) -> tuple[Tensor, Tensor]: ...
@overload
def add(
    input: Tensor | Number | _complex,
    other: Tensor | Number | _complex,
    *,
    alpha: Number | _complex | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def add(self: Tensor, alpha: Number | _complex, other: Tensor) -> Tensor: ...
@overload
def add(
    self: Tensor, alpha: Number | _complex, other: Tensor, *, out: Tensor
) -> Tensor: ...
@overload
def addbmm(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    batch1: Tensor,
    batch2: Tensor,
) -> Tensor: ...
@overload
def addbmm(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    batch1: Tensor,
    batch2: Tensor,
    *,
    out: Tensor,
) -> Tensor: ...
@overload
def addbmm(
    input: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: Number | _complex = ...,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def addbmm(
    beta: Number | _complex, self: Tensor, batch1: Tensor, batch2: Tensor
) -> Tensor: ...
@overload
def addbmm(
    beta: Number | _complex,
    self: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    out: Tensor,
) -> Tensor: ...
@overload
def addcdiv(
    self: Tensor, value: Number | _complex, tensor1: Tensor, tensor2: Tensor
) -> Tensor: ...
@overload
def addcdiv(
    self: Tensor,
    value: Number | _complex,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    out: Tensor,
) -> Tensor: ...
@overload
def addcdiv(
    input: Tensor,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    value: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def addcmul(
    self: Tensor, value: Number | _complex, tensor1: Tensor, tensor2: Tensor
) -> Tensor: ...
@overload
def addcmul(
    self: Tensor,
    value: Number | _complex,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    out: Tensor,
) -> Tensor: ...
@overload
def addcmul(
    input: Tensor,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    value: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def addmm(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    mat1: Tensor,
    mat2: Tensor,
) -> Tensor: ...
@overload
def addmm(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    mat1: Tensor,
    mat2: Tensor,
    *,
    out: Tensor,
) -> Tensor: ...
@overload
def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: Number | _complex = ...,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    out_dtype: _dtype,
    *,
    beta: Number | _complex = ...,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def addmm(
    beta: Number | _complex, self: Tensor, mat1: Tensor, mat2: Tensor
) -> Tensor: ...
@overload
def addmm(
    beta: Number | _complex, self: Tensor, mat1: Tensor, mat2: Tensor, *, out: Tensor
) -> Tensor: ...
@overload
def addmv(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    mat: Tensor,
    vec: Tensor,
) -> Tensor: ...
@overload
def addmv(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    mat: Tensor,
    vec: Tensor,
    *,
    out: Tensor,
) -> Tensor: ...
@overload
def addmv(
    input: Tensor,
    mat: Tensor,
    vec: Tensor,
    *,
    beta: Number | _complex = ...,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def addmv(
    beta: Number | _complex, self: Tensor, mat: Tensor, vec: Tensor
) -> Tensor: ...
@overload
def addmv(
    beta: Number | _complex, self: Tensor, mat: Tensor, vec: Tensor, *, out: Tensor
) -> Tensor: ...
@overload
def addmv_(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    mat: Tensor,
    vec: Tensor,
) -> Tensor: ...
@overload
def addmv_(
    input: Tensor,
    mat: Tensor,
    vec: Tensor,
    *,
    beta: Number | _complex = ...,
    alpha: Number | _complex = ...,
) -> Tensor: ...
@overload
def addmv_(
    beta: Number | _complex, self: Tensor, mat: Tensor, vec: Tensor
) -> Tensor: ...
@overload
def addr(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    vec1: Tensor,
    vec2: Tensor,
) -> Tensor: ...
@overload
def addr(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    vec1: Tensor,
    vec2: Tensor,
    *,
    out: Tensor,
) -> Tensor: ...
@overload
def addr(
    input: Tensor,
    vec1: Tensor,
    vec2: Tensor,
    *,
    beta: Number | _complex = ...,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def addr(
    beta: Number | _complex, self: Tensor, vec1: Tensor, vec2: Tensor
) -> Tensor: ...
@overload
def addr(
    beta: Number | _complex, self: Tensor, vec1: Tensor, vec2: Tensor, *, out: Tensor
) -> Tensor: ...
def adjoint(input: Tensor) -> Tensor: ...
def affine_grid_generator(
    theta: Tensor, size: Sequence[_int | SymInt], align_corners: _bool
) -> Tensor: ...
def alias_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def all(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def all(
    input: Tensor,
    dim: _size | None = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def all(
    input: Tensor, dim: _int, keepdim: _bool = ..., *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def all(
    input: Tensor,
    dim: str | EllipsisType | None,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def allclose(
    input: Tensor,
    other: Tensor,
    rtol: _float = ...,
    atol: _float = ...,
    equal_nan: _bool = ...,
) -> _bool: ...
def alpha_dropout(input: Tensor, p: _float, train: _bool) -> Tensor: ...
def alpha_dropout_(input: Tensor, p: _float, train: _bool) -> Tensor: ...
def amax(
    input: Tensor,
    dim: _int | _size = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def amin(
    input: Tensor,
    dim: _int | _size = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def aminmax(
    input: Tensor,
    *,
    dim: _int | None = ...,
    keepdim: _bool = ...,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.aminmax: ...
def angle(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def any(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def any(
    input: Tensor,
    dim: _size | None = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def any(
    input: Tensor, dim: _int, keepdim: _bool = ..., *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def any(
    input: Tensor,
    dim: str | EllipsisType | None,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def arange(
    start: Number,
    end: Number,
    step: Number,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def arange(
    start: Number,
    end: Number,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def arange(
    end: Number,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def arange(
    end: Number | _complex,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def arange(
    start: Number | _complex,
    end: Number | _complex,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def arange(
    start: Number | _complex,
    end: Number | _complex,
    step: Number | _complex = ...,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def arccos(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def arccos_(input: Tensor) -> Tensor: ...
def arccosh(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def arccosh_(input: Tensor) -> Tensor: ...
def arcsin(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def arcsin_(input: Tensor) -> Tensor: ...
def arcsinh(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def arcsinh_(input: Tensor) -> Tensor: ...
def arctan(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def arctan2(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def arctan_(input: Tensor) -> Tensor: ...
def arctanh(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def arctanh_(input: Tensor) -> Tensor: ...
def argmax(
    input: Tensor,
    dim: _int | None = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def argmin(
    input: Tensor,
    dim: _int | None = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def argsort(
    input: Tensor,
    *,
    stable: _bool,
    dim: _int = ...,
    descending: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def argsort(input: Tensor, dim: _int = ..., descending: _bool = ...) -> Tensor: ...
@overload
def argsort(
    input: Tensor, dim: str | EllipsisType | None, descending: _bool = ...
) -> Tensor: ...
def argwhere(input: Tensor) -> Tensor: ...
def as_strided(
    input: Tensor,
    size: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    storage_offset: _int | SymInt | None = ...,
) -> Tensor: ...
def as_strided_(
    input: Tensor,
    size: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    storage_offset: _int | SymInt | None = ...,
) -> Tensor: ...
def as_strided_copy(
    input: Tensor,
    size: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    storage_offset: _int | SymInt | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def as_strided_scatter(
    input: Tensor,
    src: Tensor,
    size: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    storage_offset: _int | SymInt | None = ...,
) -> Tensor: ...
def as_tensor(
    data: Any, dtype: _dtype | None = ..., device: DeviceLikeType | None = ...
) -> Tensor: ...
def asarray(
    obj: Any,
    *,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    copy: _bool | None = ...,
    requires_grad: _bool = ...,
) -> Tensor: ...
def asin(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def asin_(input: Tensor) -> Tensor: ...
def asinh(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def asinh_(input: Tensor) -> Tensor: ...
def atan(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def atan2(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def atan_(input: Tensor) -> Tensor: ...
def atanh(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def atanh_(input: Tensor) -> Tensor: ...
def avg_pool1d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    ceil_mode: _bool = ...,
    count_include_pad: _bool = ...,
) -> Tensor: ...
@overload
def baddbmm(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    batch1: Tensor,
    batch2: Tensor,
) -> Tensor: ...
@overload
def baddbmm(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    batch1: Tensor,
    batch2: Tensor,
    *,
    out: Tensor,
) -> Tensor: ...
@overload
def baddbmm(
    input: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: Number | _complex = ...,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def baddbmm(
    input: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    out_dtype: _dtype,
    *,
    beta: Number | _complex = ...,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def baddbmm(
    beta: Number | _complex, self: Tensor, batch1: Tensor, batch2: Tensor
) -> Tensor: ...
@overload
def baddbmm(
    beta: Number | _complex,
    self: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    out: Tensor,
) -> Tensor: ...
@overload
def bartlett_window(
    window_length: _int,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def bartlett_window(
    window_length: _int,
    periodic: _bool,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def batch_norm(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: _bool,
    momentum: _float,
    eps: _float,
    cudnn_enabled: _bool,
) -> Tensor: ...
def batch_norm_backward_elemt(
    grad_out: Tensor,
    input: Tensor,
    mean: Tensor,
    invstd: Tensor,
    weight: Tensor | None,
    sum_dy: Tensor,
    sum_dy_xmu: Tensor,
    count: Tensor,
) -> Tensor: ...
def batch_norm_backward_reduce(
    grad_out: Tensor,
    input: Tensor,
    mean: Tensor,
    invstd: Tensor,
    weight: Tensor | None,
    input_g: _bool,
    weight_g: _bool,
    bias_g: _bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
def batch_norm_elemt(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    invstd: Tensor,
    eps: _float,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def batch_norm_gather_stats(
    input: Tensor,
    mean: Tensor,
    invstd: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    momentum: _float,
    eps: _float,
    count: _int,
) -> tuple[Tensor, Tensor]: ...
def batch_norm_gather_stats_with_counts(
    input: Tensor,
    mean: Tensor,
    invstd: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    momentum: _float,
    eps: _float,
    counts: Tensor,
) -> tuple[Tensor, Tensor]: ...
def batch_norm_stats(input: Tensor, eps: _float) -> tuple[Tensor, Tensor]: ...
def batch_norm_update_stats(
    input: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    momentum: _float,
) -> tuple[Tensor, Tensor]: ...
@overload
def bernoulli(
    input: Tensor, *, generator: Generator | None = ..., out: Tensor | None = ...
) -> Tensor: ...
@overload
def bernoulli(
    input: Tensor, p: _float, *, generator: Generator | None = ...
) -> Tensor: ...
def bilinear(
    input1: Tensor, input2: Tensor, weight: Tensor, bias: Tensor | None = ...
) -> Tensor: ...
def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = ...,
    pos_weight: Tensor | None = ...,
    reduction: _int = ...,
) -> Tensor: ...
def bincount(
    input: Tensor, weights: Tensor | None = ..., minlength: _int | SymInt = ...
) -> Tensor: ...
def binomial(
    count: Tensor, prob: Tensor, generator: Generator | None = ...
) -> Tensor: ...
@overload
def bitwise_and(
    input: Tensor, other: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def bitwise_and(self: Number | _complex, other: Tensor) -> Tensor: ...
@overload
def bitwise_and(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def bitwise_left_shift(
    input: Tensor, other: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def bitwise_left_shift(self: Number | _complex, other: Tensor) -> Tensor: ...
@overload
def bitwise_left_shift(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def bitwise_not(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def bitwise_or(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def bitwise_or(self: Number | _complex, other: Tensor) -> Tensor: ...
@overload
def bitwise_or(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def bitwise_right_shift(
    input: Tensor, other: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def bitwise_right_shift(self: Number | _complex, other: Tensor) -> Tensor: ...
@overload
def bitwise_right_shift(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def bitwise_xor(
    input: Tensor, other: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def bitwise_xor(self: Number | _complex, other: Tensor) -> Tensor: ...
@overload
def bitwise_xor(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def blackman_window(
    window_length: _int,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def blackman_window(
    window_length: _int,
    periodic: _bool,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def bmm(input: Tensor, mat2: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def bmm(
    input: Tensor, mat2: Tensor, out_dtype: _dtype, *, out: Tensor | None = ...
) -> Tensor: ...
def broadcast_to(input: Tensor, size: Sequence[_int | SymInt]) -> Tensor: ...
@overload
def bucketize(
    input: Tensor,
    boundaries: Tensor,
    *,
    out_int32: _bool = ...,
    right: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def bucketize(
    self: Number | _complex,
    boundaries: Tensor,
    *,
    out_int32: _bool = ...,
    right: _bool = ...,
) -> Tensor: ...
def can_cast(from_: _dtype, to: _dtype) -> _bool: ...
@overload
def cat(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    dim: _int = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def cat(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    dim: str | EllipsisType | None,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def ccol_indices_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def ceil(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def ceil_(input: Tensor) -> Tensor: ...
def celu(input: Tensor, alpha: Number | _complex = ...) -> Tensor: ...
def celu_(input: Tensor, alpha: Number | _complex = ...) -> Tensor: ...
def channel_shuffle(input: Tensor, groups: _int | SymInt) -> Tensor: ...
def cholesky(
    input: Tensor, upper: _bool = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def cholesky_inverse(
    input: Tensor, upper: _bool = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def cholesky_solve(
    input: Tensor, input2: Tensor, upper: _bool = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def choose_qparams_optimized(
    input: Tensor, numel: _int, n_bins: _int, ratio: _float, bit_width: _int
) -> tuple[Tensor, Tensor]: ...
def chunk(input: Tensor, chunks: _int, dim: _int = ...) -> tuple[Tensor, ...]: ...
@overload
def clamp(
    input: Tensor,
    min: Tensor | None = ...,
    max: Tensor | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def clamp(
    input: Tensor,
    min: Number | _complex | None = ...,
    max: Number | _complex | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def clamp_(
    input: Tensor, min: Tensor | None = ..., max: Tensor | None = ...
) -> Tensor: ...
@overload
def clamp_(
    input: Tensor,
    min: Number | _complex | None = ...,
    max: Number | _complex | None = ...,
) -> Tensor: ...
@overload
def clamp_max(input: Tensor, max: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def clamp_max(
    input: Tensor, max: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def clamp_max_(input: Tensor, max: Tensor) -> Tensor: ...
@overload
def clamp_max_(input: Tensor, max: Number | _complex) -> Tensor: ...
@overload
def clamp_min(input: Tensor, min: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def clamp_min(
    input: Tensor, min: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def clamp_min_(input: Tensor, min: Tensor) -> Tensor: ...
@overload
def clamp_min_(input: Tensor, min: Number | _complex) -> Tensor: ...
@overload
def clip(
    input: Tensor,
    min: Tensor | None = ...,
    max: Tensor | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def clip(
    input: Tensor,
    min: Number | _complex | None = ...,
    max: Number | _complex | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def clip_(
    input: Tensor, min: Tensor | None = ..., max: Tensor | None = ...
) -> Tensor: ...
@overload
def clip_(
    input: Tensor,
    min: Number | _complex | None = ...,
    max: Number | _complex | None = ...,
) -> Tensor: ...
def clone(input: Tensor, *, memory_format: memory_format | None = ...) -> Tensor: ...
def col_indices_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def column_stack(
    tensors: tuple[Tensor, ...] | list[Tensor] | None, *, out: Tensor | None = ...
) -> Tensor: ...
def combinations(
    input: Tensor, r: _int = ..., with_replacement: _bool = ...
) -> Tensor: ...
def complex(real: Tensor, imag: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def concat(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    dim: _int = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def concat(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    dim: str | EllipsisType | None,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def concatenate(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    dim: _int = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def concatenate(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    dim: str | EllipsisType | None,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def conj(input: Tensor) -> Tensor: ...
def conj_physical(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def conj_physical_(input: Tensor) -> Tensor: ...
def constant_pad_nd(
    input: Tensor, pad: Sequence[_int | SymInt], value: Number | _complex = ...
) -> Tensor: ...
@overload
def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _int | SymInt | Sequence[_int | SymInt] = ...,
    padding: _int | SymInt | Sequence[_int | SymInt] = ...,
    dilation: _int | SymInt | Sequence[_int | SymInt] = ...,
    groups: _int | SymInt = ...,
) -> Tensor: ...
@overload
def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _int | SymInt | Sequence[_int | SymInt] = ...,
    padding: str = ...,
    dilation: _int | SymInt | Sequence[_int | SymInt] = ...,
    groups: _int | SymInt = ...,
) -> Tensor: ...
@overload
def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _int | SymInt | Sequence[_int | SymInt] = ...,
    padding: _int | SymInt | Sequence[_int | SymInt] = ...,
    dilation: _int | SymInt | Sequence[_int | SymInt] = ...,
    groups: _int | SymInt = ...,
) -> Tensor: ...
@overload
def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _int | SymInt | Sequence[_int | SymInt] = ...,
    padding: str = ...,
    dilation: _int | SymInt | Sequence[_int | SymInt] = ...,
    groups: _int | SymInt = ...,
) -> Tensor: ...
@overload
def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _int | SymInt | Sequence[_int | SymInt] = ...,
    padding: _int | SymInt | Sequence[_int | SymInt] = ...,
    dilation: _int | SymInt | Sequence[_int | SymInt] = ...,
    groups: _int | SymInt = ...,
) -> Tensor: ...
@overload
def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _int | SymInt | Sequence[_int | SymInt] = ...,
    padding: str = ...,
    dilation: _int | SymInt | Sequence[_int | SymInt] = ...,
    groups: _int | SymInt = ...,
) -> Tensor: ...
def conv_tbc(
    input: Tensor, weight: Tensor, bias: Tensor, pad: _int = ...
) -> Tensor: ...
def conv_transpose1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _int | SymInt | Sequence[_int | SymInt] = ...,
    padding: _int | SymInt | Sequence[_int | SymInt] = ...,
    output_padding: _int | SymInt | Sequence[_int | SymInt] = ...,
    groups: _int | SymInt = ...,
    dilation: _int | SymInt | Sequence[_int | SymInt] = ...,
) -> Tensor: ...
def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _int | SymInt | Sequence[_int | SymInt] = ...,
    padding: _int | SymInt | Sequence[_int | SymInt] = ...,
    output_padding: _int | SymInt | Sequence[_int | SymInt] = ...,
    groups: _int | SymInt = ...,
    dilation: _int | SymInt | Sequence[_int | SymInt] = ...,
) -> Tensor: ...
def conv_transpose3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _int | SymInt | Sequence[_int | SymInt] = ...,
    padding: _int | SymInt | Sequence[_int | SymInt] = ...,
    output_padding: _int | SymInt | Sequence[_int | SymInt] = ...,
    groups: _int | SymInt = ...,
    dilation: _int | SymInt | Sequence[_int | SymInt] = ...,
) -> Tensor: ...
def convolution(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Sequence[_int | SymInt],
    padding: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    transposed: _bool,
    output_padding: Sequence[_int | SymInt],
    groups: _int | SymInt,
) -> Tensor: ...
@overload
def copysign(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def copysign(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def corrcoef(input: Tensor) -> Tensor: ...
def cos(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def cos_(input: Tensor) -> Tensor: ...
def cosh(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def cosh_(input: Tensor) -> Tensor: ...
def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: _float = ...,
    reduction: _int = ...,
) -> Tensor: ...
def cosine_similarity(
    x1: Tensor, x2: Tensor, dim: _int = ..., eps: _float = ...
) -> Tensor: ...
@overload
def count_nonzero(input: Tensor, dim: _int | None = ...) -> Tensor: ...
@overload
def count_nonzero(input: Tensor, dim: _size) -> Tensor: ...
def cov(
    input: Tensor,
    *,
    correction: _int = ...,
    fweights: Tensor | None = ...,
    aweights: Tensor | None = ...,
) -> Tensor: ...
def cross(
    input: Tensor, other: Tensor, dim: _int | None = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def crow_indices_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: _size,
    target_lengths: _size,
    blank: _int = ...,
    reduction: _int = ...,
    zero_infinity: _bool = ...,
) -> Tensor: ...
@overload
def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: _int = ...,
    reduction: _int = ...,
    zero_infinity: _bool = ...,
) -> Tensor: ...
def cudnn_affine_grid_generator(
    theta: Tensor, N: _int, C: _int, H: _int, W: _int
) -> Tensor: ...
def cudnn_batch_norm(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: _bool,
    exponential_average_factor: _float,
    epsilon: _float,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
def cudnn_convolution(
    input: Tensor,
    weight: Tensor,
    padding: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
    benchmark: _bool,
    deterministic: _bool,
    allow_tf32: _bool,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def cudnn_convolution_add_relu(
    input: Tensor,
    weight: Tensor,
    z: Tensor,
    alpha: Number | _complex | None,
    bias: Tensor | None,
    stride: Sequence[_int | SymInt],
    padding: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
) -> Tensor: ...
def cudnn_convolution_relu(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Sequence[_int | SymInt],
    padding: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
) -> Tensor: ...
def cudnn_convolution_transpose(
    input: Tensor,
    weight: Tensor,
    padding: Sequence[_int | SymInt],
    output_padding: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
    benchmark: _bool,
    deterministic: _bool,
    allow_tf32: _bool,
) -> Tensor: ...
def cudnn_grid_sampler(input: Tensor, grid: Tensor) -> Tensor: ...
def cudnn_is_acceptable(input: Tensor) -> _bool: ...
@overload
def cummax(
    input: Tensor,
    dim: _int,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.cummax: ...
@overload
def cummax(
    input: Tensor,
    dim: str | EllipsisType | None,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.cummax: ...
@overload
def cummin(
    input: Tensor,
    dim: _int,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.cummin: ...
@overload
def cummin(
    input: Tensor,
    dim: str | EllipsisType | None,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.cummin: ...
@overload
def cumprod(
    input: Tensor, dim: _int, *, dtype: _dtype | None = ..., out: Tensor | None = ...
) -> Tensor: ...
@overload
def cumprod(
    input: Tensor,
    dim: str | EllipsisType | None,
    *,
    dtype: _dtype | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def cumsum(
    input: Tensor, dim: _int, *, dtype: _dtype | None = ..., out: Tensor | None = ...
) -> Tensor: ...
@overload
def cumsum(
    input: Tensor,
    dim: str | EllipsisType | None,
    *,
    dtype: _dtype | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def cumulative_trapezoid(y: Tensor, x: Tensor, *, dim: _int = ...) -> Tensor: ...
@overload
def cumulative_trapezoid(
    y: Tensor, *, dx: Number | _complex = ..., dim: _int = ...
) -> Tensor: ...
def deg2rad(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def deg2rad_(input: Tensor) -> Tensor: ...
@overload
def dequantize(input: Tensor) -> Tensor: ...
@overload
def dequantize(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def det(input: Tensor) -> Tensor: ...
def detach(input: Tensor) -> Tensor: ...
def detach_(input: Tensor) -> Tensor: ...
def detach_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def diag(
    input: Tensor, diagonal: _int = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def diag_embed(
    input: Tensor, offset: _int = ..., dim1: _int = ..., dim2: _int = ...
) -> Tensor: ...
def diagflat(input: Tensor, offset: _int = ...) -> Tensor: ...
@overload
def diagonal(
    input: Tensor, offset: _int = ..., dim1: _int = ..., dim2: _int = ...
) -> Tensor: ...
@overload
def diagonal(
    input: Tensor,
    *,
    outdim: str | EllipsisType | None,
    dim1: str | EllipsisType | None,
    dim2: str | EllipsisType | None,
    offset: _int = ...,
) -> Tensor: ...
def diagonal_copy(
    input: Tensor,
    offset: _int = ...,
    dim1: _int = ...,
    dim2: _int = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def diagonal_scatter(
    input: Tensor, src: Tensor, offset: _int = ..., dim1: _int = ..., dim2: _int = ...
) -> Tensor: ...
def diff(
    input: Tensor,
    n: _int = ...,
    dim: _int = ...,
    prepend: Tensor | None = ...,
    append: Tensor | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def digamma(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def dist(input: Tensor, other: Tensor, p: Number | _complex = ...) -> Tensor: ...
def div(
    input: Tensor | Number,
    other: Tensor | Number,
    *,
    rounding_mode: str | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def divide(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def divide(
    input: Tensor, other: Tensor, *, rounding_mode: str | None, out: Tensor | None = ...
) -> Tensor: ...
@overload
def divide(
    input: Tensor, other: Number | _complex, *, rounding_mode: str | None
) -> Tensor: ...
@overload
def divide(input: Tensor, other: Number | _complex) -> Tensor: ...
def dot(input: Tensor, tensor: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def dropout(input: Tensor, p: _float, train: _bool) -> Tensor: ...
def dropout_(input: Tensor, p: _float, train: _bool) -> Tensor: ...
def dsmm(input: Tensor, mat2: Tensor) -> Tensor: ...
@overload
def dsplit(input: Tensor, sections: _int) -> tuple[Tensor, ...]: ...
@overload
def dsplit(input: Tensor, indices: _size) -> tuple[Tensor, ...]: ...
def dstack(
    tensors: tuple[Tensor, ...] | list[Tensor] | None, *, out: Tensor | None = ...
) -> Tensor: ...
def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: _int | SymInt = ...,
    scale_grad_by_freq: _bool = ...,
    sparse: _bool = ...,
) -> Tensor: ...
@overload
def embedding_bag(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: _bool,
    mode: _int,
    sparse: _bool,
    per_sample_weights: Tensor | None,
    include_last_offset: _bool,
    padding_idx: _int | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
@overload
def embedding_bag(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: _bool = ...,
    mode: _int = ...,
    sparse: _bool = ...,
    per_sample_weights: Tensor | None = ...,
    include_last_offset: _bool = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
def embedding_renorm_(
    input: Tensor, indices: Tensor, max_norm: _float, norm_type: _float
) -> Tensor: ...
@overload
def empty(
    size: Sequence[_int | SymInt],
    *,
    memory_format: memory_format | None = ...,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def empty(
    *size: _int | SymInt,
    memory_format: memory_format | None = ...,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def empty(
    size: _size,
    *,
    names: Sequence[str | EllipsisType | None] | None,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def empty(
    *size: _int,
    names: Sequence[str | EllipsisType | None] | None,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def empty_like(
    input: Tensor,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def empty_permuted(
    size: Sequence[_int | SymInt],
    physical_layout: _size,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def empty_quantized(
    size: _size,
    qtensor: Tensor,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def empty_strided(
    size: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def eq(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def eq(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def equal(input: Tensor, other: Tensor) -> _bool: ...
def erf(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def erf_(input: Tensor) -> Tensor: ...
def erfc(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def erfc_(input: Tensor) -> Tensor: ...
def erfinv(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def exp(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def exp2(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def exp2_(input: Tensor) -> Tensor: ...
def exp_(input: Tensor) -> Tensor: ...
def expand_copy(
    input: Tensor,
    size: Sequence[_int | SymInt],
    *,
    implicit: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def expm1(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def expm1_(input: Tensor) -> Tensor: ...
@overload
def eye(
    n: _int | SymInt,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def eye(
    n: _int | SymInt,
    m: _int | SymInt,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def fake_quantize_per_channel_affine(
    input: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    axis: _int,
    quant_min: _int,
    quant_max: _int,
) -> Tensor: ...
@overload
def fake_quantize_per_tensor_affine(
    input: Tensor, scale: _float, zero_point: _int, quant_min: _int, quant_max: _int
) -> Tensor: ...
@overload
def fake_quantize_per_tensor_affine(
    input: Tensor, scale: Tensor, zero_point: Tensor, quant_min: _int, quant_max: _int
) -> Tensor: ...
@overload
def fbgemm_linear_fp16_weight(
    input: Tensor, packed_weight: Tensor, bias: Tensor
) -> Tensor: ...
@overload
def fbgemm_linear_fp16_weight(
    input: Tensor, packed_weight: Tensor, bias: Tensor, output: Tensor
) -> Tensor: ...
@overload
def fbgemm_linear_fp16_weight_fp32_activation(
    input: Tensor, packed_weight: Tensor, bias: Tensor | None
) -> Tensor: ...
@overload
def fbgemm_linear_fp16_weight_fp32_activation(
    input: Tensor, packed_weight: Tensor, bias: Tensor | None, output: Tensor
) -> Tensor: ...
def fbgemm_linear_int8_weight(
    input: Tensor,
    weight: Tensor,
    packed: Tensor,
    col_offsets: Tensor,
    weight_scale: Number | _complex,
    weight_zero_point: Number | _complex,
    bias: Tensor,
) -> Tensor: ...
def fbgemm_linear_int8_weight_fp32_activation(
    input: Tensor,
    weight: Tensor,
    packed: Tensor,
    col_offsets: Tensor,
    weight_scale: Number | _complex,
    weight_zero_point: Number | _complex,
    bias: Tensor,
) -> Tensor: ...
def fbgemm_linear_quantize_weight(
    input: Tensor,
) -> tuple[Tensor, Tensor, _float, _int]: ...
def fbgemm_pack_gemm_matrix_fp16(input: Tensor) -> Tensor: ...
@overload
def fbgemm_pack_quantized_matrix(input: Tensor) -> Tensor: ...
@overload
def fbgemm_pack_quantized_matrix(input: Tensor, K: _int, N: _int) -> Tensor: ...
def feature_alpha_dropout(input: Tensor, p: _float, train: _bool) -> Tensor: ...
def feature_alpha_dropout_(input: Tensor, p: _float, train: _bool) -> Tensor: ...
def feature_dropout(input: Tensor, p: _float, train: _bool) -> Tensor: ...
def feature_dropout_(input: Tensor, p: _float, train: _bool) -> Tensor: ...
@overload
def fill(input: Tensor, value: Tensor) -> Tensor: ...
@overload
def fill(input: Tensor, value: Number | _complex) -> Tensor: ...
@overload
def fill_(input: Tensor, value: Tensor) -> Tensor: ...
@overload
def fill_(input: Tensor, value: Number | _complex) -> Tensor: ...
def fix(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def fix_(input: Tensor) -> Tensor: ...
@overload
def flatten(input: Tensor, start_dim: _int = ..., end_dim: _int = ...) -> Tensor: ...
@overload
def flatten(
    input: Tensor, start_dim: _int, end_dim: _int, out_dim: str | EllipsisType | None
) -> Tensor: ...
@overload
def flatten(
    input: Tensor,
    start_dim: str | EllipsisType | None,
    end_dim: str | EllipsisType | None,
    out_dim: str | EllipsisType | None,
) -> Tensor: ...
@overload
def flatten(
    input: Tensor,
    dims: Sequence[str | EllipsisType | None],
    out_dim: str | EllipsisType | None,
) -> Tensor: ...
def flip(input: Tensor, dims: _size) -> Tensor: ...
def fliplr(input: Tensor) -> Tensor: ...
def flipud(input: Tensor) -> Tensor: ...
@overload
def float_power(
    input: Tensor, exponent: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def float_power(
    self: Number | _complex, exponent: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def float_power(
    input: Tensor, exponent: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def floor(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def floor_(input: Tensor) -> Tensor: ...
def floor_divide(
    input: Tensor | Number, other: Tensor | Number, *, out: Tensor | None = ...
) -> Tensor: ...
def fmax(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def fmin(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def fmod(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def fmod(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def frac(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def frac_(input: Tensor) -> Tensor: ...
def frexp(
    input: Tensor, *, out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...
) -> torch.return_types.frexp: ...
def frobenius_norm(
    input: Tensor, dim: _int | _size, keepdim: _bool = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def from_file(
    filename: str,
    shared: _bool | None = ...,
    size: _int | None = ...,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def from_numpy(ndarray: Any) -> Tensor: ...
def frombuffer(
    buffer: Any,
    *,
    dtype: _dtype,
    count: int = ...,
    offset: int = ...,
    requires_grad: _bool = ...,
) -> Tensor: ...
@overload
def full(
    size: _size,
    fill_value: Number | _complex,
    *,
    out: Tensor | None = ...,
    layout: _layout = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def full(
    size: _size,
    fill_value: Number | _complex,
    *,
    names: list[str | None],
    layout: _layout = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def full(
    size: Sequence[_int | SymInt],
    fill_value: Number | _complex,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def full(
    size: _size,
    fill_value: Number | _complex,
    *,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def full_like(
    input: Tensor,
    fill_value: Number | _complex,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def fused_moving_avg_obs_fake_quant(
    input: Tensor,
    observer_on: Tensor,
    fake_quant_on: Tensor,
    running_min: Tensor,
    running_max: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    averaging_const: _float,
    quant_min: _int,
    quant_max: _int,
    ch_axis: _int,
    per_row_fake_quant: _bool = ...,
    symmetric_quant: _bool = ...,
) -> Tensor: ...
@overload
def gather(
    input: Tensor,
    dim: _int,
    index: Tensor,
    *,
    sparse_grad: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def gather(
    input: Tensor,
    dim: str | EllipsisType | None,
    index: Tensor,
    *,
    sparse_grad: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def gcd(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def gcd_(input: Tensor, other: Tensor) -> Tensor: ...
@overload
def ge(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def ge(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def geqrf(
    input: Tensor, *, out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...
) -> torch.return_types.geqrf: ...
def ger(input: Tensor, vec2: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def get_default_dtype() -> _dtype: ...
def get_num_interop_threads() -> _int: ...
def get_num_threads() -> _int: ...
@overload
def gradient(
    input: Tensor,
    *,
    spacing: Number | _complex | None = ...,
    dim: _int | None = ...,
    edge_order: _int = ...,
) -> tuple[Tensor, ...]: ...
@overload
def gradient(
    input: Tensor,
    *,
    spacing: Sequence[Number | _complex],
    dim: _int | None = ...,
    edge_order: _int = ...,
) -> tuple[Tensor, ...]: ...
@overload
def gradient(
    input: Tensor,
    *,
    spacing: Sequence[Number | _complex],
    dim: _size,
    edge_order: _int = ...,
) -> tuple[Tensor, ...]: ...
@overload
def gradient(
    input: Tensor,
    *,
    spacing: tuple[Tensor, ...] | list[Tensor] | None,
    dim: _int | None = ...,
    edge_order: _int = ...,
) -> tuple[Tensor, ...]: ...
@overload
def gradient(
    input: Tensor, *, spacing: Number | _complex, dim: _size, edge_order: _int = ...
) -> tuple[Tensor, ...]: ...
@overload
def gradient(
    input: Tensor,
    *,
    spacing: tuple[Tensor, ...] | list[Tensor] | None,
    dim: _size,
    edge_order: _int = ...,
) -> tuple[Tensor, ...]: ...
@overload
def gradient(
    input: Tensor, *, dim: _size, edge_order: _int = ...
) -> tuple[Tensor, ...]: ...
@overload
def greater(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def greater(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def greater_equal(
    input: Tensor, other: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def greater_equal(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def grid_sampler(
    input: Tensor,
    grid: Tensor,
    interpolation_mode: _int,
    padding_mode: _int,
    align_corners: _bool,
) -> Tensor: ...
def grid_sampler_2d(
    input: Tensor,
    grid: Tensor,
    interpolation_mode: _int,
    padding_mode: _int,
    align_corners: _bool,
) -> Tensor: ...
def grid_sampler_3d(
    input: Tensor,
    grid: Tensor,
    interpolation_mode: _int,
    padding_mode: _int,
    align_corners: _bool,
) -> Tensor: ...
def group_norm(
    input: Tensor,
    num_groups: _int,
    weight: Tensor | None = ...,
    bias: Tensor | None = ...,
    eps: _float = ...,
    cudnn_enabled: _bool = ...,
) -> Tensor: ...
@overload
def gru(
    data: Tensor,
    batch_sizes: Tensor,
    hx: Tensor,
    params: tuple[Tensor, ...] | list[Tensor] | None,
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
) -> tuple[Tensor, Tensor]: ...
@overload
def gru(
    input: Tensor,
    hx: Tensor,
    params: tuple[Tensor, ...] | list[Tensor] | None,
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    batch_first: _bool,
) -> tuple[Tensor, Tensor]: ...
def gru_cell(
    input: Tensor,
    hx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor | None = ...,
    b_hh: Tensor | None = ...,
) -> Tensor: ...
@overload
def gt(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def gt(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def hamming_window(
    window_length: _int,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def hamming_window(
    window_length: _int,
    periodic: _bool,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def hamming_window(
    window_length: _int,
    periodic: _bool,
    alpha: _float,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def hamming_window(
    window_length: _int,
    periodic: _bool,
    alpha: _float,
    beta: _float,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def hann_window(
    window_length: _int,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def hann_window(
    window_length: _int,
    periodic: _bool,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def hardshrink(
    input: Tensor, lambd: Number | _complex = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def hash_tensor(
    input: Tensor,
    dim: _int | _size = ...,
    *,
    keepdim: _bool = ...,
    mode: _int = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def heaviside(input: Tensor, values: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def hinge_embedding_loss(
    input: Tensor, target: Tensor, margin: _float = ..., reduction: _int = ...
) -> Tensor: ...
def histc(
    input: Tensor,
    bins: _int = ...,
    min: Number | _complex = ...,
    max: Number | _complex = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def histogram(
    input: Tensor,
    bins: Tensor,
    *,
    weight: Tensor | None = ...,
    density: _bool = ...,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.histogram: ...
@overload
def histogram(
    input: Tensor,
    bins: _int = ...,
    *,
    range: Sequence[_float] | None = ...,
    weight: Tensor | None = ...,
    density: _bool = ...,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.histogram: ...
@overload
def histogramdd(
    input: Tensor,
    bins: _int,
    range: Sequence[_float] | None = ...,
    weight: Tensor | None = ...,
    density: _bool = ...,
) -> torch.return_types.histogramdd: ...
@overload
def histogramdd(
    input: Tensor,
    bins: _size,
    range: Sequence[_float] | None = ...,
    weight: Tensor | None = ...,
    density: _bool = ...,
) -> torch.return_types.histogramdd: ...
@overload
def histogramdd(
    input: Tensor,
    bins: tuple[Tensor, ...] | list[Tensor] | None,
    range: Sequence[_float] | None = ...,
    weight: Tensor | None = ...,
    density: _bool = ...,
) -> torch.return_types.histogramdd: ...
def hsmm(input: Tensor, mat2: Tensor) -> Tensor: ...
@overload
def hsplit(input: Tensor, sections: _int) -> tuple[Tensor, ...]: ...
@overload
def hsplit(input: Tensor, indices: _size) -> tuple[Tensor, ...]: ...
def hspmm(mat1: Tensor, mat2: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def hstack(
    tensors: tuple[Tensor, ...] | list[Tensor] | None, *, out: Tensor | None = ...
) -> Tensor: ...
def hypot(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def i0(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def i0_(input: Tensor) -> Tensor: ...
def igamma(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def igammac(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def imag(input: Tensor) -> Tensor: ...
@overload
def index_add(
    input: Tensor,
    dim: _int,
    index: Tensor,
    source: Tensor,
    *,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def index_add(
    input: Tensor,
    dim: str | EllipsisType | None,
    index: Tensor,
    source: Tensor,
    *,
    alpha: Number | _complex = ...,
) -> Tensor: ...
@overload
def index_copy(
    input: Tensor, dim: _int, index: Tensor, source: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def index_copy(
    input: Tensor, dim: str | EllipsisType | None, index: Tensor, source: Tensor
) -> Tensor: ...
@overload
def index_fill(input: Tensor, dim: _int, index: Tensor, value: Tensor) -> Tensor: ...
@overload
def index_fill(
    input: Tensor, dim: str | EllipsisType | None, index: Tensor, value: Tensor
) -> Tensor: ...
@overload
def index_fill(
    input: Tensor, dim: _int, index: Tensor, value: Number | _complex
) -> Tensor: ...
@overload
def index_fill(
    input: Tensor,
    dim: str | EllipsisType | None,
    index: Tensor,
    value: Number | _complex,
) -> Tensor: ...
def index_put(
    input: Tensor,
    indices: tuple[Tensor, ...] | list[Tensor] | None,
    values: Tensor,
    accumulate: _bool = ...,
) -> Tensor: ...
def index_put_(
    input: Tensor,
    indices: tuple[Tensor, ...] | list[Tensor] | None,
    values: Tensor,
    accumulate: _bool = ...,
) -> Tensor: ...
def index_reduce(
    input: Tensor,
    dim: _int,
    index: Tensor,
    source: Tensor,
    reduce: str,
    *,
    include_self: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def index_select(
    input: Tensor, dim: _int, index: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def index_select(
    input: Tensor,
    dim: str | EllipsisType | None,
    index: Tensor,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def indices_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def init_num_threads() -> None: ...
def inner(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def instance_norm(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    use_input_stats: _bool,
    momentum: _float,
    eps: _float,
    cudnn_enabled: _bool,
) -> Tensor: ...
def int_repr(input: Tensor) -> Tensor: ...
def inverse(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def is_complex(input: Tensor) -> _bool: ...
def is_conj(input: Tensor) -> _bool: ...
def is_distributed(input: Tensor) -> _bool: ...
def is_floating_point(input: Tensor) -> _bool: ...
def is_grad_enabled() -> _bool: ...
def is_inference(input: Tensor) -> _bool: ...
def is_inference_mode_enabled() -> _bool: ...
def is_neg(input: Tensor) -> _bool: ...
def is_nonzero(input: Tensor) -> _bool: ...
def is_same_size(input: Tensor, other: Tensor) -> _bool: ...
def is_signed(input: Tensor) -> _bool: ...
def is_vulkan_available() -> _bool: ...
def isclose(
    input: Tensor,
    other: Tensor,
    rtol: _float = ...,
    atol: _float = ...,
    equal_nan: _bool = ...,
) -> Tensor: ...
def isfinite(input: Tensor) -> Tensor: ...
@overload
def isin(
    elements: Tensor,
    test_elements: Tensor,
    *,
    assume_unique: _bool = ...,
    invert: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def isin(
    element: Number | _complex,
    test_elements: Tensor,
    *,
    assume_unique: _bool = ...,
    invert: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def isin(
    elements: Tensor,
    test_element: Number | _complex,
    *,
    assume_unique: _bool = ...,
    invert: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def isinf(input: Tensor) -> Tensor: ...
def isnan(input: Tensor) -> Tensor: ...
def isneginf(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def isposinf(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def isreal(input: Tensor) -> Tensor: ...
def istft(
    input: Tensor,
    n_fft: _int,
    hop_length: _int | None = ...,
    win_length: _int | None = ...,
    window: Tensor | None = ...,
    center: _bool = ...,
    normalized: _bool = ...,
    onesided: _bool | None = ...,
    length: _int | None = ...,
    return_complex: _bool = ...,
) -> Tensor: ...
@overload
def kaiser_window(
    window_length: _int,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def kaiser_window(
    window_length: _int,
    periodic: _bool,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def kaiser_window(
    window_length: _int,
    periodic: _bool,
    beta: _float,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def kl_div(
    input: Tensor, target: Tensor, reduction: _int = ..., *, log_target: _bool = ...
) -> Tensor: ...
def kron(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def kthvalue(
    input: Tensor,
    k: _int | SymInt,
    dim: _int = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.kthvalue: ...
@overload
def kthvalue(
    input: Tensor,
    k: _int | SymInt,
    dim: str | EllipsisType | None,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.kthvalue: ...
def layer_norm(
    input: Tensor,
    normalized_shape: Sequence[_int | SymInt],
    weight: Tensor | None = ...,
    bias: Tensor | None = ...,
    eps: _float = ...,
    cudnn_enable: _bool = ...,
) -> Tensor: ...
def lcm(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def lcm_(input: Tensor, other: Tensor) -> Tensor: ...
def ldexp(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def ldexp_(input: Tensor, other: Tensor) -> Tensor: ...
@overload
def le(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def le(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def lerp(
    input: Tensor, end: Tensor, weight: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def lerp(
    input: Tensor, end: Tensor, weight: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def less(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def less(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def less_equal(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def less_equal(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def lgamma(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def linspace(
    start: Number,
    end: Number,
    steps: _int | None = ...,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def linspace(
    start: Tensor,
    end: Tensor,
    steps: _int,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def linspace(
    start: Number | _complex,
    end: Tensor,
    steps: _int,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def linspace(
    start: Tensor,
    end: Number | _complex,
    steps: _int,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def linspace(
    start: Number | _complex,
    end: Number | _complex,
    steps: _int,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def log(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def log10(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def log10_(input: Tensor) -> Tensor: ...
def log1p(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def log1p_(input: Tensor) -> Tensor: ...
def log2(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def log2_(input: Tensor) -> Tensor: ...
def log_(input: Tensor) -> Tensor: ...
def log_softmax(
    input: Tensor,
    dim: _int | Sequence[_int],
    dtype: _dtype | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def logaddexp(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def logaddexp2(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def logcumsumexp(input: Tensor, dim: _int, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def logcumsumexp(
    input: Tensor, dim: str | EllipsisType | None, *, out: Tensor | None = ...
) -> Tensor: ...
def logdet(input: Tensor) -> Tensor: ...
def logical_and(
    input: Tensor, other: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
def logical_not(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def logical_or(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def logical_xor(
    input: Tensor, other: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
def logit(
    input: Tensor, eps: _float | None = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def logit_(input: Tensor, eps: _float | None = ...) -> Tensor: ...
@overload
def logspace(
    start: Number,
    end: Number,
    steps: _int | None = ...,
    base: _float = ...,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def logspace(
    start: Tensor,
    end: Tensor,
    steps: _int,
    base: _float = ...,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def logspace(
    start: Number | _complex,
    end: Tensor,
    steps: _int,
    base: _float = ...,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def logspace(
    start: Tensor,
    end: Number | _complex,
    steps: _int,
    base: _float = ...,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def logspace(
    start: Number | _complex,
    end: Number | _complex,
    steps: _int,
    base: _float = ...,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def logsumexp(
    input: Tensor,
    dim: _int | Sequence[_int],
    keepdim: bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def lstm(
    data: Tensor,
    batch_sizes: Tensor,
    hx: tuple[Tensor, ...] | list[Tensor] | None,
    params: tuple[Tensor, ...] | list[Tensor] | None,
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
) -> tuple[Tensor, Tensor, Tensor]: ...
@overload
def lstm(
    input: Tensor,
    hx: tuple[Tensor, ...] | list[Tensor] | None,
    params: tuple[Tensor, ...] | list[Tensor] | None,
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    batch_first: _bool,
) -> tuple[Tensor, Tensor, Tensor]: ...
def lstm_cell(
    input: Tensor,
    hx: tuple[Tensor, ...] | list[Tensor] | None,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor | None = ...,
    b_hh: Tensor | None = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def lt(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def lt(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def lu_solve(
    input: Tensor, LU_data: Tensor, LU_pivots: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
def lu_unpack(
    LU_data: Tensor,
    LU_pivots: Tensor,
    unpack_data: _bool = ...,
    unpack_pivots: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.lu_unpack: ...
def margin_ranking_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: _float = ...,
    reduction: _int = ...,
) -> Tensor: ...
@overload
def masked_fill(input: Tensor, mask: Tensor, value: Tensor) -> Tensor: ...
@overload
def masked_fill(input: Tensor, mask: Tensor, value: Number | _complex) -> Tensor: ...
def masked_scatter(input: Tensor, mask: Tensor, source: Tensor) -> Tensor: ...
def masked_select(
    input: Tensor, mask: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
def matmul(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def matrix_exp(input: Tensor) -> Tensor: ...
def matrix_power(input: Tensor, n: _int, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def max(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def max(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def max(
    input: Tensor,
    dim: _int,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.max: ...
@overload
def max(
    input: Tensor,
    dim: str | EllipsisType | None,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.max: ...
def max_pool1d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: _bool = ...,
) -> Tensor: ...
def max_pool1d_with_indices(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: _bool = ...,
) -> tuple[Tensor, Tensor]: ...
def max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: _bool = ...,
) -> Tensor: ...
def max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: _bool = ...,
) -> Tensor: ...
def maximum(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def mean(
    input: Tensor,
    dim: _int | Sequence[_int] | None = ...,
    keepdim: bool = ...,
    *,
    dtype: _dtype | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def median(input: Tensor) -> Tensor: ...
@overload
def median(
    input: Tensor,
    dim: _int,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.median: ...
@overload
def median(
    input: Tensor,
    dim: str | EllipsisType | None,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.median: ...
@overload
def min(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def min(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def min(
    input: Tensor,
    dim: _int,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.min: ...
@overload
def min(
    input: Tensor,
    dim: str | EllipsisType | None,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.min: ...
def minimum(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def miopen_batch_norm(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: _bool,
    exponential_average_factor: _float,
    epsilon: _float,
) -> tuple[Tensor, Tensor, Tensor]: ...
def miopen_convolution(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    padding: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
    benchmark: _bool,
    deterministic: _bool,
) -> Tensor: ...
def miopen_convolution_add_relu(
    input: Tensor,
    weight: Tensor,
    z: Tensor,
    alpha: Number | _complex | None,
    bias: Tensor | None,
    stride: Sequence[_int | SymInt],
    padding: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
) -> Tensor: ...
def miopen_convolution_relu(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Sequence[_int | SymInt],
    padding: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
) -> Tensor: ...
def miopen_convolution_transpose(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    padding: Sequence[_int | SymInt],
    output_padding: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
    benchmark: _bool,
    deterministic: _bool,
) -> Tensor: ...
def miopen_depthwise_convolution(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    padding: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
    benchmark: _bool,
    deterministic: _bool,
) -> Tensor: ...
def miopen_rnn(
    input: Tensor,
    weight: tuple[Tensor, ...] | list[Tensor] | None,
    weight_stride0: _int,
    hx: Tensor,
    cx: Tensor | None,
    mode: _int,
    hidden_size: _int,
    num_layers: _int,
    batch_first: _bool,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    batch_sizes: _size,
    dropout_state: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
def mkldnn_adaptive_avg_pool2d(
    input: Tensor, output_size: _int | _size, *, out: Tensor | None = ...
) -> Tensor: ...
def mkldnn_convolution(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    padding: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
) -> Tensor: ...
def mkldnn_linear_backward_weights(
    grad_output: Tensor, input: Tensor, weight: Tensor, bias_defined: _bool
) -> tuple[Tensor, Tensor]: ...
def mkldnn_max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: _bool = ...,
) -> Tensor: ...
def mkldnn_max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: _bool = ...,
) -> Tensor: ...
def mkldnn_rnn_layer(
    input: Tensor,
    weight0: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    weight3: Tensor,
    hx_: Tensor,
    cx_: Tensor,
    reverse: _bool,
    batch_sizes: _size,
    mode: _int,
    hidden_size: _int,
    num_layers: _int,
    has_biases: _bool,
    bidirectional: _bool,
    batch_first: _bool,
    train: _bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
@overload
def mm(input: Tensor, mat2: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def mm(
    input: Tensor, mat2: Tensor, out_dtype: _dtype, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def mode(
    input: Tensor,
    dim: _int = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.mode: ...
@overload
def mode(
    input: Tensor,
    dim: str | EllipsisType | None,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.mode: ...
@overload
def moveaxis(input: Tensor, source: _int, destination: _int) -> Tensor: ...
@overload
def moveaxis(input: Tensor, source: _size, destination: _size) -> Tensor: ...
@overload
def movedim(input: Tensor, source: _int, destination: _int) -> Tensor: ...
@overload
def movedim(input: Tensor, source: _size, destination: _size) -> Tensor: ...
def msort(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def mul(
    input: Tensor | Number | _complex,
    other: Tensor | Number | _complex,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def multinomial(
    input: Tensor,
    num_samples: _int | SymInt,
    replacement: _bool = ...,
    *,
    generator: Generator | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def multiply(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def multiply(input: Tensor, other: Number | _complex) -> Tensor: ...
def mv(input: Tensor, vec: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def mvlgamma(input: Tensor, p: _int, *, out: Tensor | None = ...) -> Tensor: ...
def nan_to_num(
    input: Tensor,
    nan: _float | None = ...,
    posinf: _float | None = ...,
    neginf: _float | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def nan_to_num_(
    input: Tensor,
    nan: _float | None = ...,
    posinf: _float | None = ...,
    neginf: _float | None = ...,
) -> Tensor: ...
def nanmean(
    input: Tensor,
    dim: _int | _size | None = ...,
    keepdim: _bool = ...,
    *,
    dtype: _dtype | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def nanmedian(input: Tensor) -> Tensor: ...
@overload
def nanmedian(
    input: Tensor,
    dim: _int,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.nanmedian: ...
@overload
def nanmedian(
    input: Tensor,
    dim: str | EllipsisType | None,
    keepdim: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.nanmedian: ...
@overload
def nanquantile(
    input: Tensor,
    q: Tensor,
    dim: _int | None = ...,
    keepdim: _bool = ...,
    *,
    interpolation: str = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def nanquantile(
    input: Tensor,
    q: _float,
    dim: _int | None = ...,
    keepdim: _bool = ...,
    *,
    interpolation: str = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def nansum(
    input: Tensor,
    dim: _int | _size | None = ...,
    keepdim: _bool = ...,
    *,
    dtype: _dtype | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def narrow(
    input: Tensor, dim: _int, start: Tensor, length: _int | SymInt
) -> Tensor: ...
@overload
def narrow(
    input: Tensor, dim: _int, start: _int | SymInt, length: _int | SymInt
) -> Tensor: ...
def narrow_copy(
    input: Tensor,
    dim: _int,
    start: _int | SymInt,
    length: _int | SymInt,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def native_batch_norm(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: _bool,
    momentum: _float,
    eps: _float,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
def native_channel_shuffle(input: Tensor, groups: _int | SymInt) -> Tensor: ...
def native_dropout(
    input: Tensor, p: _float, train: _bool | None
) -> tuple[Tensor, Tensor]: ...
def native_group_norm(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    N: _int | SymInt,
    C: _int | SymInt,
    HxW: _int | SymInt,
    group: _int,
    eps: _float,
) -> tuple[Tensor, Tensor, Tensor]: ...
def native_layer_norm(
    input: Tensor,
    normalized_shape: Sequence[_int | SymInt],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: _float,
) -> tuple[Tensor, Tensor, Tensor]: ...
@overload
def native_norm(
    input: Tensor,
    p: Number | _complex | None,
    dim: _int | _size,
    keepdim: _bool,
    dtype: _dtype | None,
) -> Tensor: ...
@overload
def native_norm(input: Tensor, p: Number | _complex = ...) -> Tensor: ...
@overload
def ne(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def ne(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def neg(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def neg_(input: Tensor) -> Tensor: ...
def negative(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def negative_(input: Tensor) -> Tensor: ...
def nextafter(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def nonzero(
    input: Tensor, *, as_tuple: Literal[False] = ..., out: Tensor | None = ...
) -> Tensor: ...
@overload
def nonzero(input: Tensor, *, as_tuple: Literal[True]) -> tuple[Tensor, ...]: ...
def nonzero_static(
    input: Tensor,
    *,
    size: _int | SymInt,
    fill_value: _int = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def norm_except_dim(v: Tensor, pow: _int = ..., dim: _int = ...) -> Tensor: ...
@overload
def normal(
    mean: Tensor,
    std: Tensor,
    *,
    generator: Generator | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def normal(
    mean: Tensor,
    std: _float = ...,
    *,
    generator: Generator | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def normal(
    mean: _float,
    std: Tensor,
    *,
    generator: Generator | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def normal(
    mean: _float,
    std: _float,
    size: Sequence[_int | SymInt],
    *,
    generator: Generator | None = ...,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def not_equal(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def not_equal(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def nuclear_norm(
    input: Tensor, dim: _int | _size, keepdim: _bool = ..., *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def nuclear_norm(
    input: Tensor, keepdim: _bool = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def numel(self: Tensor) -> _int: ...
@overload
def ones(
    size: Sequence[_int | SymInt],
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def ones(
    *size: _int | SymInt,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def ones(
    size: _size,
    *,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def ones(
    *size: _int,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def ones_like(
    input: Tensor,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def orgqr(input: Tensor, input2: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def ormqr(
    input: Tensor,
    input2: Tensor,
    input3: Tensor,
    left: _bool = ...,
    transpose: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def outer(input: Tensor, vec2: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def pairwise_distance(
    x1: Tensor, x2: Tensor, p: _float = ..., eps: _float = ..., keepdim: _bool = ...
) -> Tensor: ...
def pdist(input: Tensor, p: _float = ...) -> Tensor: ...
def permute(input: Tensor, dims: _size) -> Tensor: ...
def permute_copy(input: Tensor, dims: _size, *, out: Tensor | None = ...) -> Tensor: ...
def pinverse(input: Tensor, rcond: _float = ...) -> Tensor: ...
def pixel_shuffle(input: Tensor, upscale_factor: _int) -> Tensor: ...
def pixel_unshuffle(input: Tensor, downscale_factor: _int) -> Tensor: ...
def poisson(input: Tensor, generator: Generator | None = ...) -> Tensor: ...
def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    log_input: _bool,
    full: _bool,
    eps: _float,
    reduction: _int,
) -> Tensor: ...
def polar(abs: Tensor, angle: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def polygamma(n: _int, input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def positive(input: Tensor) -> Tensor: ...
@overload
def pow(input: Tensor, exponent: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def pow(
    self: Number | _complex, exponent: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def pow(
    input: Tensor, exponent: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def prelu(input: Tensor, weight: Tensor) -> Tensor: ...
@overload
def prod(input: Tensor, *, dtype: _dtype | None = ...) -> Tensor: ...
@overload
def prod(
    input: Tensor,
    dim: _int,
    keepdim: _bool = ...,
    *,
    dtype: _dtype | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def prod(
    input: Tensor,
    dim: str | EllipsisType | None,
    keepdim: _bool = ...,
    *,
    dtype: _dtype | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def promote_types(type1: _dtype, type2: _dtype) -> _dtype: ...
def put(
    input: Tensor, index: Tensor, source: Tensor, accumulate: _bool = ...
) -> Tensor: ...
def q_per_channel_axis(input: Tensor) -> _int: ...
def q_per_channel_scales(input: Tensor) -> Tensor: ...
def q_per_channel_zero_points(input: Tensor) -> Tensor: ...
def q_scale(input: Tensor) -> _float: ...
def q_zero_point(input: Tensor) -> _int: ...
def qr(
    input: Tensor,
    some: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.qr: ...
@overload
def quantile(
    input: Tensor,
    q: Tensor,
    dim: _int | None = ...,
    keepdim: _bool = ...,
    *,
    interpolation: str = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def quantile(
    input: Tensor,
    q: _float,
    dim: _int | None = ...,
    keepdim: _bool = ...,
    *,
    interpolation: str = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def quantize_per_channel(
    input: Tensor, scales: Tensor, zero_points: Tensor, axis: _int, dtype: _dtype
) -> Tensor: ...
@overload
def quantize_per_tensor(
    input: Tensor, scale: Tensor, zero_point: Tensor, dtype: _dtype
) -> Tensor: ...
@overload
def quantize_per_tensor(
    input: Tensor, scale: _float, zero_point: _int, dtype: _dtype
) -> Tensor: ...
@overload
def quantize_per_tensor(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    scales: Tensor,
    zero_points: Tensor,
    dtype: _dtype,
) -> tuple[Tensor, ...]: ...
def quantize_per_tensor_dynamic(
    input: Tensor, dtype: _dtype, reduce_range: _bool
) -> Tensor: ...
def quantized_batch_norm(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    var: Tensor,
    eps: _float,
    output_scale: _float,
    output_zero_point: _int,
) -> Tensor: ...
def quantized_gru_cell(
    input: Tensor,
    hx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    packed_ih: Tensor,
    packed_hh: Tensor,
    col_offsets_ih: Tensor,
    col_offsets_hh: Tensor,
    scale_ih: Number | _complex,
    scale_hh: Number | _complex,
    zero_point_ih: Number | _complex,
    zero_point_hh: Number | _complex,
) -> Tensor: ...
def quantized_lstm_cell(
    input: Tensor,
    hx: tuple[Tensor, ...] | list[Tensor] | None,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    packed_ih: Tensor,
    packed_hh: Tensor,
    col_offsets_ih: Tensor,
    col_offsets_hh: Tensor,
    scale_ih: Number | _complex,
    scale_hh: Number | _complex,
    zero_point_ih: Number | _complex,
    zero_point_hh: Number | _complex,
) -> tuple[Tensor, Tensor]: ...
def quantized_max_pool1d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: _bool = ...,
) -> Tensor: ...
def quantized_max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: _bool = ...,
) -> Tensor: ...
def quantized_max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: _bool = ...,
) -> Tensor: ...
def quantized_rnn_relu_cell(
    input: Tensor,
    hx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    packed_ih: Tensor,
    packed_hh: Tensor,
    col_offsets_ih: Tensor,
    col_offsets_hh: Tensor,
    scale_ih: Number | _complex,
    scale_hh: Number | _complex,
    zero_point_ih: Number | _complex,
    zero_point_hh: Number | _complex,
) -> Tensor: ...
def quantized_rnn_tanh_cell(
    input: Tensor,
    hx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    packed_ih: Tensor,
    packed_hh: Tensor,
    col_offsets_ih: Tensor,
    col_offsets_hh: Tensor,
    scale_ih: Number | _complex,
    scale_hh: Number | _complex,
    zero_point_ih: Number | _complex,
    zero_point_hh: Number | _complex,
) -> Tensor: ...
def rad2deg(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def rad2deg_(input: Tensor) -> Tensor: ...
@overload
def rand(
    size: Sequence[_int | SymInt],
    *,
    generator: Generator | None,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def rand(
    *size: _int | SymInt,
    generator: Generator | None,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def rand(
    size: Sequence[_int | SymInt],
    *,
    generator: Generator | None,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def rand(
    *size: _int | SymInt,
    generator: Generator | None,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def rand(
    size: Sequence[_int | SymInt],
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def rand(
    *size: _int | SymInt,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def rand(
    size: Sequence[_int | SymInt],
    *,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def rand(
    *size: _int | SymInt,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def rand_like(
    input: Tensor,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randint(
    low: _int,
    high: _int,
    size: _size,
    *,
    generator: Generator | None = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def randint(
    high: _int,
    size: _size,
    *,
    generator: Generator | None = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def randint(
    high: _int | SymInt,
    size: Sequence[_int | SymInt],
    *,
    generator: Generator | None,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randint(
    high: _int | SymInt,
    size: Sequence[_int | SymInt],
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randint(
    low: _int | SymInt,
    high: _int | SymInt,
    size: Sequence[_int | SymInt],
    *,
    generator: Generator | None,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randint(
    low: _int | SymInt,
    high: _int | SymInt,
    size: Sequence[_int | SymInt],
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randint_like(
    input: Tensor,
    low: _int | SymInt,
    high: _int | SymInt,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randint_like(
    input: Tensor,
    high: Tensor,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randint_like(
    input: Tensor,
    high: _int | SymInt,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randn(
    size: Sequence[_int | SymInt],
    *,
    generator: Generator | None,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randn(
    *size: _int | SymInt,
    generator: Generator | None,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randn(
    size: Sequence[_int | SymInt],
    *,
    generator: Generator | None,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randn(
    *size: _int | SymInt,
    generator: Generator | None,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randn(
    size: Sequence[_int | SymInt],
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randn(
    *size: _int | SymInt,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randn(
    size: Sequence[_int | SymInt],
    *,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randn(
    *size: _int | SymInt,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def randn_like(
    input: Tensor,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randperm(
    n: _int | SymInt,
    *,
    generator: Generator | None,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def randperm(
    n: _int | SymInt,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def range(
    start: Number,
    end: Number,
    step: Number = ...,
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
def ravel(input: Tensor) -> Tensor: ...
def real(input: Tensor) -> Tensor: ...
def reciprocal(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def reciprocal_(input: Tensor) -> Tensor: ...
def relu(input: Tensor) -> Tensor: ...
def relu_(input: Tensor) -> Tensor: ...
@overload
def remainder(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def remainder(self: Number | _complex, other: Tensor) -> Tensor: ...
@overload
def remainder(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
def renorm(
    input: Tensor,
    p: Number | _complex,
    dim: _int,
    maxnorm: Number | _complex,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def repeat_interleave(
    input: Tensor,
    repeats: Tensor,
    dim: _int | None = ...,
    *,
    output_size: _int | SymInt | None = ...,
) -> Tensor: ...
@overload
def repeat_interleave(
    repeats: Tensor, *, output_size: _int | SymInt | None = ...
) -> Tensor: ...
@overload
def repeat_interleave(
    input: Tensor,
    repeats: _int | SymInt,
    dim: _int | None = ...,
    *,
    output_size: _int | SymInt | None = ...,
) -> Tensor: ...
def reshape(input: Tensor, shape: Sequence[_int | SymInt]) -> Tensor: ...
def resize_as_(
    input: Tensor, the_template: Tensor, *, memory_format: memory_format | None = ...
) -> Tensor: ...
def resize_as_sparse_(input: Tensor, the_template: Tensor) -> Tensor: ...
def resolve_conj(input: Tensor) -> Tensor: ...
def resolve_neg(input: Tensor) -> Tensor: ...
@overload
def result_type(tensor: Tensor, other: Tensor) -> _dtype: ...
@overload
def result_type(scalar: Number | _complex, tensor: Tensor) -> _dtype: ...
@overload
def result_type(tensor: Tensor, other: Number | _complex) -> _dtype: ...
@overload
def result_type(scalar1: Number | _complex, scalar2: Number | _complex) -> _dtype: ...
def rms_norm(
    input: Tensor,
    normalized_shape: Sequence[_int | SymInt],
    weight: Tensor | None = ...,
    eps: _float | None = ...,
) -> Tensor: ...
@overload
def rnn_relu(
    data: Tensor,
    batch_sizes: Tensor,
    hx: Tensor,
    params: tuple[Tensor, ...] | list[Tensor] | None,
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
) -> tuple[Tensor, Tensor]: ...
@overload
def rnn_relu(
    input: Tensor,
    hx: Tensor,
    params: tuple[Tensor, ...] | list[Tensor] | None,
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    batch_first: _bool,
) -> tuple[Tensor, Tensor]: ...
def rnn_relu_cell(
    input: Tensor,
    hx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor | None = ...,
    b_hh: Tensor | None = ...,
) -> Tensor: ...
@overload
def rnn_tanh(
    data: Tensor,
    batch_sizes: Tensor,
    hx: Tensor,
    params: tuple[Tensor, ...] | list[Tensor] | None,
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
) -> tuple[Tensor, Tensor]: ...
@overload
def rnn_tanh(
    input: Tensor,
    hx: Tensor,
    params: tuple[Tensor, ...] | list[Tensor] | None,
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    batch_first: _bool,
) -> tuple[Tensor, Tensor]: ...
def rnn_tanh_cell(
    input: Tensor,
    hx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor | None = ...,
    b_hh: Tensor | None = ...,
) -> Tensor: ...
def roll(
    input: Tensor,
    shifts: _int | SymInt | Sequence[_int | SymInt],
    dims: _int | _size = ...,
) -> Tensor: ...
def rot90(input: Tensor, k: _int = ..., dims: _size = ...) -> Tensor: ...
@overload
def round(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def round(input: Tensor, *, decimals: _int, out: Tensor | None = ...) -> Tensor: ...
@overload
def round_(input: Tensor) -> Tensor: ...
@overload
def round_(input: Tensor, *, decimals: _int) -> Tensor: ...
def row_indices_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def row_stack(
    tensors: tuple[Tensor, ...] | list[Tensor] | None, *, out: Tensor | None = ...
) -> Tensor: ...
def rrelu(
    input: Tensor,
    lower: Number | _complex = ...,
    upper: Number | _complex = ...,
    training: _bool = ...,
    generator: Generator | None = ...,
) -> Tensor: ...
def rrelu_(
    input: Tensor,
    lower: Number | _complex = ...,
    upper: Number | _complex = ...,
    training: _bool = ...,
    generator: Generator | None = ...,
) -> Tensor: ...
def rsqrt(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def rsqrt_(input: Tensor) -> Tensor: ...
@overload
def rsub(input: Tensor, other: Tensor, *, alpha: Number | _complex = ...) -> Tensor: ...
@overload
def rsub(
    input: Tensor, other: Number | _complex, alpha: Number | _complex = ...
) -> Tensor: ...
def saddmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: Number = ...,
    alpha: Number = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def scalar_tensor(
    s: Number | _complex,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def scatter(
    input: Tensor,
    dim: _int,
    index: Tensor,
    src: Tensor,
    *,
    reduce: str,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def scatter(
    input: Tensor, dim: _int, index: Tensor, src: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def scatter(
    input: Tensor,
    dim: _int,
    index: Tensor,
    value: Number | _complex,
    *,
    reduce: str,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def scatter(
    input: Tensor, dim: str | EllipsisType | None, index: Tensor, src: Tensor
) -> Tensor: ...
@overload
def scatter(
    input: Tensor,
    dim: _int,
    index: Tensor,
    value: Number | _complex,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def scatter(
    input: Tensor,
    dim: str | EllipsisType | None,
    index: Tensor,
    value: Number | _complex,
) -> Tensor: ...
@overload
def scatter_add(
    input: Tensor, dim: _int, index: Tensor, src: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def scatter_add(
    input: Tensor, dim: str | EllipsisType | None, index: Tensor, src: Tensor
) -> Tensor: ...
def scatter_reduce(
    input: Tensor,
    dim: _int,
    index: Tensor,
    src: Tensor,
    reduce: str,
    *,
    include_self: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def searchsorted(
    sorted_sequence: Tensor,
    input: Tensor,
    *,
    out_int32: _bool = ...,
    right: _bool = ...,
    side: str | None = ...,
    sorter: Tensor | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def searchsorted(
    sorted_sequence: Tensor,
    self: Number | _complex,
    *,
    out_int32: _bool = ...,
    right: _bool = ...,
    side: str | None = ...,
    sorter: Tensor | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def segment_reduce(
    data: Tensor,
    reduce: str,
    *,
    lengths: Tensor | None = ...,
    indices: Tensor | None = ...,
    offsets: Tensor | None = ...,
    axis: _int = ...,
    unsafe: _bool = ...,
    initial: Number | _complex | None = ...,
) -> Tensor: ...
@overload
def select(input: Tensor, dim: _int, index: _int | SymInt) -> Tensor: ...
@overload
def select(input: Tensor, dim: str | EllipsisType | None, index: _int) -> Tensor: ...
def select_copy(
    input: Tensor, dim: _int, index: _int | SymInt, *, out: Tensor | None = ...
) -> Tensor: ...
def select_scatter(
    input: Tensor, src: Tensor, dim: _int, index: _int | SymInt
) -> Tensor: ...
def selu(input: Tensor) -> Tensor: ...
def selu_(input: Tensor) -> Tensor: ...
def set_flush_denormal(mode: _bool) -> _bool: ...
def set_num_interop_threads(num: _int) -> None: ...
def set_num_threads(num: _int) -> None: ...
def sgn(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def sigmoid(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def sigmoid_(input: Tensor) -> Tensor: ...
def sign(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def signbit(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def sin(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def sin_(input: Tensor) -> Tensor: ...
def sinc(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def sinc_(input: Tensor) -> Tensor: ...
def sinh(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def sinh_(input: Tensor) -> Tensor: ...
def slice_copy(
    input: Tensor,
    dim: _int = ...,
    start: _int | SymInt | None = ...,
    end: _int | SymInt | None = ...,
    step: _int | SymInt = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def slice_inverse(
    input: Tensor,
    src: Tensor,
    dim: _int = ...,
    start: _int | SymInt | None = ...,
    end: _int | SymInt | None = ...,
    step: _int | SymInt = ...,
) -> Tensor: ...
def slice_scatter(
    input: Tensor,
    src: Tensor,
    dim: _int = ...,
    start: _int | SymInt | None = ...,
    end: _int | SymInt | None = ...,
    step: _int | SymInt = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def slogdet(
    input: Tensor, *, out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...
) -> torch.return_types.slogdet: ...
def smm(input: Tensor, mat2: Tensor) -> Tensor: ...
def softmax(
    input: Tensor,
    dim: _int | Sequence[_int],
    dtype: _dtype | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def sort(
    input: Tensor,
    *,
    stable: _bool | None,
    dim: _int = ...,
    descending: _bool = ...,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.sort: ...
@overload
def sort(
    input: Tensor,
    dim: _int = ...,
    descending: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.sort: ...
@overload
def sort(
    input: Tensor,
    *,
    stable: _bool | None,
    dim: str | EllipsisType | None,
    descending: _bool = ...,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.sort: ...
@overload
def sort(
    input: Tensor,
    dim: str | EllipsisType | None,
    descending: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.sort: ...
def sparse_bsc_tensor(
    ccol_indices: Tensor | list,
    row_indices: Tensor | list,
    values: Tensor | list,
    size: _size | None = ...,
    *,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    check_invariants: _bool | None = ...,
) -> Tensor: ...
def sparse_bsr_tensor(
    crow_indices: Tensor | list,
    col_indices: Tensor | list,
    values: Tensor | list,
    size: _size | None = ...,
    *,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    check_invariants: _bool | None = ...,
) -> Tensor: ...
def sparse_compressed_tensor(
    compressed_indices: Tensor | list,
    plain_indices: Tensor | list,
    values: Tensor | list,
    size: _size | None = ...,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    check_invariants: _bool | None = ...,
) -> Tensor: ...
def sparse_coo_tensor(
    indices: Tensor,
    values: Tensor | list,
    size: _size | None = ...,
    *,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    check_invariants: _bool | None = ...,
    is_coalesced: _bool | None = ...,
) -> Tensor: ...
def sparse_csc_tensor(
    ccol_indices: Tensor | list,
    row_indices: Tensor | list,
    values: Tensor | list,
    size: _size | None = ...,
    *,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    check_invariants: _bool | None = ...,
) -> Tensor: ...
def sparse_csr_tensor(
    crow_indices: Tensor | list,
    col_indices: Tensor | list,
    values: Tensor | list,
    size: _size | None = ...,
    *,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    check_invariants: _bool | None = ...,
) -> Tensor: ...
def split_copy(
    input: Tensor,
    split_size: _int | SymInt,
    dim: _int = ...,
    *,
    out: tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> None: ...
def split_with_sizes(
    input: Tensor, split_sizes: Sequence[_int | SymInt], dim: _int = ...
) -> tuple[Tensor, ...]: ...
def split_with_sizes_copy(
    input: Tensor,
    split_sizes: Sequence[_int | SymInt],
    dim: _int = ...,
    *,
    out: tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> None: ...
def spmm(input: Tensor, mat2: Tensor) -> Tensor: ...
def sqrt(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def sqrt_(input: Tensor) -> Tensor: ...
def square(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def square_(input: Tensor) -> Tensor: ...
@overload
def squeeze(input: Tensor) -> Tensor: ...
@overload
def squeeze(input: Tensor, dim: _int) -> Tensor: ...
@overload
def squeeze(input: Tensor, dim: _size) -> Tensor: ...
@overload
def squeeze(input: Tensor, dim: str | EllipsisType | None) -> Tensor: ...
@overload
def squeeze_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def squeeze_copy(input: Tensor, dim: _int, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def squeeze_copy(input: Tensor, dim: _size, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def sspaddmm(
    beta: Number | _complex,
    self: Tensor,
    alpha: Number | _complex,
    mat1: Tensor,
    mat2: Tensor,
) -> Tensor: ...
@overload
def sspaddmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: Number | _complex = ...,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def sspaddmm(
    beta: Number | _complex, self: Tensor, mat1: Tensor, mat2: Tensor
) -> Tensor: ...
def stack(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    dim: _int = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def std(
    input: Tensor,
    dim: _int | Sequence[_int] | None = ...,
    keepdim: bool = ...,
    *,
    correction: int | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def std_mean(
    input: Tensor, dim: _int | _size | None, unbiased: _bool = ..., keepdim: _bool = ...
) -> tuple[Tensor, Tensor]: ...
@overload
def std_mean(
    input: Tensor,
    dim: _int | _size | None = ...,
    *,
    correction: Number | _complex | None = ...,
    keepdim: _bool = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def std_mean(input: Tensor, unbiased: _bool = ...) -> tuple[Tensor, Tensor]: ...
@overload
def std_mean(
    input: Tensor,
    dim: Sequence[str | EllipsisType | None],
    *,
    correction: Number | _complex | None = ...,
    keepdim: _bool = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def std_mean(
    input: Tensor,
    dim: Sequence[str | EllipsisType | None],
    unbiased: _bool = ...,
    keepdim: _bool = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def sub(
    input: Tensor | Number | _complex,
    other: Tensor | Number | _complex,
    *,
    alpha: Number | _complex | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def sub(self: Tensor, alpha: Number | _complex, other: Tensor) -> Tensor: ...
@overload
def sub(
    self: Tensor, alpha: Number | _complex, other: Tensor, *, out: Tensor
) -> Tensor: ...
@overload
def subtract(
    input: Tensor,
    other: Tensor,
    *,
    alpha: Number | _complex = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def subtract(
    input: Tensor, other: Number | _complex, alpha: Number | _complex = ...
) -> Tensor: ...
def sum(
    input: Tensor,
    dim: _int | Sequence[_int] | None = ...,
    keepdim: bool = ...,
    *,
    dtype: _dtype | None = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
def svd(
    input: Tensor,
    some: _bool = ...,
    compute_uv: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.svd: ...
def swapaxes(input: Tensor, axis0: _int, axis1: _int) -> Tensor: ...
def swapdims(input: Tensor, dim0: _int, dim1: _int) -> Tensor: ...
def sym_constrain_range(
    size: Number | _complex, *, min: _int | None = ..., max: _int | None = ...
) -> None: ...
def sym_constrain_range_for_size(
    size: Number | _complex, *, min: _int | None = ..., max: _int | None = ...
) -> None: ...
def t(input: Tensor) -> Tensor: ...
def t_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def take(input: Tensor, index: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def take_along_dim(
    input: Tensor, indices: Tensor, dim: _int | None = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def tan(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def tan_(input: Tensor) -> Tensor: ...
def tanh(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def tanh_(input: Tensor) -> Tensor: ...
def tensor(
    data: Any,
    dtype: _dtype | None = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: _bool = ...,
    pin_memory: _bool = ...,
) -> Tensor: ...
@overload
def tensor_split(
    input: Tensor, tensor_indices_or_sections: Tensor, dim: _int = ...
) -> tuple[Tensor, ...]: ...
@overload
def tensor_split(
    input: Tensor, sections: _int | SymInt, dim: _int = ...
) -> tuple[Tensor, ...]: ...
@overload
def tensor_split(
    input: Tensor, indices: Sequence[_int | SymInt], dim: _int = ...
) -> tuple[Tensor, ...]: ...
def threshold(
    input: Tensor,
    threshold: Number | _complex,
    value: Number | _complex,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def threshold_(
    input: Tensor, threshold: Number | _complex, value: Number | _complex
) -> Tensor: ...
def tile(input: Tensor, dims: Sequence[_int | SymInt]) -> Tensor: ...
def topk(
    input: Tensor,
    k: _int | SymInt,
    dim: _int = ...,
    largest: _bool = ...,
    sorted: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.topk: ...
def trace(input: Tensor) -> Tensor: ...
@overload
def transpose(input: Tensor, dim0: _int, dim1: _int) -> Tensor: ...
@overload
def transpose(
    input: Tensor, dim0: str | EllipsisType | None, dim1: str | EllipsisType | None
) -> Tensor: ...
def transpose_copy(
    input: Tensor, dim0: _int, dim1: _int, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def trapezoid(y: Tensor, x: Tensor, *, dim: _int = ...) -> Tensor: ...
@overload
def trapezoid(y: Tensor, *, dx: Number | _complex = ..., dim: _int = ...) -> Tensor: ...
@overload
def trapz(y: Tensor, *, dx: _float = ..., dim: _int = ...) -> Tensor: ...
@overload
def trapz(y: Tensor, x: Tensor, *, dim: _int = ...) -> Tensor: ...
def triangular_solve(
    input: Tensor,
    A: Tensor,
    upper: _bool = ...,
    transpose: _bool = ...,
    unitriangular: _bool = ...,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> torch.return_types.triangular_solve: ...
def tril(
    input: Tensor, diagonal: _int = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def tril_indices(
    row: _int,
    col: _int,
    offset: _int = ...,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: _float = ...,
    p: _float = ...,
    eps: _float = ...,
    swap: _bool = ...,
    reduction: _int = ...,
) -> Tensor: ...
def triu(
    input: Tensor, diagonal: _int = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def triu_indices(
    row: _int,
    col: _int,
    offset: _int = ...,
    *,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def true_divide(
    input: Tensor | Number, other: Tensor | Number, *, out: Tensor | None = ...
) -> Tensor: ...
def trunc(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def trunc_(input: Tensor) -> Tensor: ...
@overload
def unbind(input: Tensor, dim: _int = ...) -> tuple[Tensor, ...]: ...
@overload
def unbind(input: Tensor, dim: str | EllipsisType | None) -> tuple[Tensor, ...]: ...
def unbind_copy(
    input: Tensor,
    dim: _int = ...,
    *,
    out: tuple[Tensor, ...] | list[Tensor] | None = ...,
) -> None: ...
@overload
def unflatten(
    input: Tensor,
    dim: str | EllipsisType | None,
    sizes: Sequence[_int | SymInt],
    names: Sequence[str | EllipsisType | None],
) -> Tensor: ...
@overload
def unflatten(input: Tensor, dim: _int, sizes: Sequence[_int | SymInt]) -> Tensor: ...
def unfold_copy(
    input: Tensor, dimension: _int, size: _int, step: _int, *, out: Tensor | None = ...
) -> Tensor: ...
def unique_dim(
    input: Tensor,
    dim: _int,
    sorted: _bool = ...,
    return_inverse: _bool = ...,
    return_counts: _bool = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
def unsafe_chunk(
    input: Tensor, chunks: _int, dim: _int = ...
) -> tuple[Tensor, ...]: ...
def unsafe_split(
    input: Tensor, split_size: _int | SymInt, dim: _int = ...
) -> tuple[Tensor, ...]: ...
def unsafe_split_with_sizes(
    input: Tensor, split_sizes: Sequence[_int | SymInt], dim: _int = ...
) -> tuple[Tensor, ...]: ...
def unsqueeze(input: Tensor, dim: _int) -> Tensor: ...
def unsqueeze_copy(input: Tensor, dim: _int, *, out: Tensor | None = ...) -> Tensor: ...
def values_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def vander(x: Tensor, N: _int | None = ..., increasing: _bool = ...) -> Tensor: ...
@overload
def var(
    input: Tensor,
    dim: _int | _size | None,
    unbiased: _bool = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def var(
    input: Tensor,
    dim: _int | _size | None = ...,
    *,
    correction: Number | _complex | None = ...,
    keepdim: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def var(input: Tensor, unbiased: _bool = ...) -> Tensor: ...
@overload
def var(
    input: Tensor,
    dim: Sequence[str | EllipsisType | None],
    *,
    correction: Number | _complex | None = ...,
    keepdim: _bool = ...,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def var(
    input: Tensor,
    dim: Sequence[str | EllipsisType | None],
    unbiased: _bool = ...,
    keepdim: _bool = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
@overload
def var_mean(
    input: Tensor, dim: _int | _size | None, unbiased: _bool = ..., keepdim: _bool = ...
) -> tuple[Tensor, Tensor]: ...
@overload
def var_mean(
    input: Tensor,
    dim: _int | _size | None = ...,
    *,
    correction: Number | _complex | None = ...,
    keepdim: _bool = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def var_mean(input: Tensor, unbiased: _bool = ...) -> tuple[Tensor, Tensor]: ...
@overload
def var_mean(
    input: Tensor,
    dim: Sequence[str | EllipsisType | None],
    *,
    correction: Number | _complex | None = ...,
    keepdim: _bool = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def var_mean(
    input: Tensor,
    dim: Sequence[str | EllipsisType | None],
    unbiased: _bool = ...,
    keepdim: _bool = ...,
) -> tuple[Tensor, Tensor]: ...
def vdot(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def view_as_complex(input: Tensor) -> Tensor: ...
def view_as_complex_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def view_as_real(input: Tensor) -> Tensor: ...
def view_as_real_copy(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def view_copy(input: Tensor, dtype: _dtype, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def view_copy(
    input: Tensor, size: Sequence[_int | SymInt], *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def vsplit(input: Tensor, sections: _int) -> tuple[Tensor, ...]: ...
@overload
def vsplit(input: Tensor, indices: _size) -> tuple[Tensor, ...]: ...
def vstack(
    tensors: tuple[Tensor, ...] | list[Tensor] | None, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def where(condition: Tensor) -> tuple[Tensor, ...]: ...
@overload
def where(
    condition: Tensor, input: Tensor, other: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def where(condition: Tensor, self: Number | _complex, other: Tensor) -> Tensor: ...
@overload
def where(condition: Tensor, input: Tensor, other: Number | _complex) -> Tensor: ...
@overload
def where(
    condition: Tensor, self: Number | _complex, other: Number | _complex
) -> Tensor: ...
@overload
def xlogy(input: Tensor, other: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
@overload
def xlogy(
    self: Number | _complex, other: Tensor, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def xlogy(
    input: Tensor, other: Number | _complex, *, out: Tensor | None = ...
) -> Tensor: ...
@overload
def xlogy_(input: Tensor, other: Tensor) -> Tensor: ...
@overload
def xlogy_(input: Tensor, other: Number | _complex) -> Tensor: ...
def zero_(input: Tensor) -> Tensor: ...
@overload
def zeros(
    size: Sequence[_int | SymInt],
    *,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def zeros(
    *size: _int | SymInt,
    out: Tensor | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def zeros(
    size: _size,
    *,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
@overload
def zeros(
    *size: _int,
    names: Sequence[str | EllipsisType | None] | None,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...
def zeros_like(
    input: Tensor,
    *,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = ...,
    layout: _layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: _bool | None = ...,
    requires_grad: _bool | None = ...,
) -> Tensor: ...

# Backfilled from the installed torch stub: these are named in __all__ but the
# generator dropped their definitions, so every call site inferred Unknown.
def _adaptive_avg_pool2d(
    input: Tensor,
    output_size: _int | SymInt | Sequence[_int | SymInt],
) -> Tensor: ...
def _adaptive_avg_pool3d(
    input: Tensor,
    output_size: _int | SymInt | Sequence[_int | SymInt],
) -> Tensor: ...
def _add_batch_dim(input: Tensor, batch_dim: _int, level: _int) -> Tensor: ...
@overload
def _add_relu(
    input: Tensor,
    other: Tensor,
    *,
    alpha: Number | _complex = 1,
    out: Tensor | None = None,
) -> Tensor: ...
@overload
def _add_relu(
    input: Tensor,
    other: Number | _complex,
    alpha: Number | _complex = 1,
) -> Tensor: ...
@overload
def _add_relu_(
    input: Tensor,
    other: Tensor,
    *,
    alpha: Number | _complex = 1,
) -> Tensor: ...
@overload
def _add_relu_(
    input: Tensor,
    other: Number | _complex,
    alpha: Number | _complex = 1,
) -> Tensor: ...
def _addmm_activation(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: Number | _complex = 1,
    alpha: Number | _complex = 1,
    use_gelu: _bool = False,
    out: Tensor | None = None,
) -> Tensor: ...
@overload
def _aminmax(input: Tensor) -> tuple[Tensor, Tensor]: ...
@overload
def _aminmax(
    input: Tensor,
    dim: _int,
    keepdim: _bool = False,
) -> tuple[Tensor, Tensor]: ...
def _amp_foreach_non_finite_check_and_unscale_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    found_inf: Tensor,
    inv_scale: Tensor,
) -> None: ...
def _amp_update_scale_(
    input: Tensor,
    growth_tracker: Tensor,
    found_inf: Tensor,
    scale_growth_factor: _float,
    scale_backoff_factor: _float,
    growth_interval: _int,
) -> Tensor: ...
@overload
def _assert_async(input: Tensor) -> None: ...
@overload
def _assert_async(input: Tensor, assert_msg: str) -> None: ...
def _assert_scalar(self: Number | _complex, assert_msg: str) -> None: ...
def _assert_tensor_metadata(
    a: Tensor,
    size: Sequence[_int | SymInt] | None = None,
    stride: Sequence[_int | SymInt] | None = None,
    dtype: _dtype | None = None,
    *,
    device: DeviceLikeType | None = None,
    layout: _layout | None = None,
) -> None: ...
def _batch_norm_impl_index(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: _bool,
    momentum: _float,
    eps: _float,
    cudnn_enabled: _bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, _int]: ...
def _cast_Byte(input: Tensor, non_blocking: _bool = False) -> Tensor: ...
def _cast_Char(input: Tensor, non_blocking: _bool = False) -> Tensor: ...
def _cast_Double(input: Tensor, non_blocking: _bool = False) -> Tensor: ...
def _cast_Float(input: Tensor, non_blocking: _bool = False) -> Tensor: ...
def _cast_Half(input: Tensor, non_blocking: _bool = False) -> Tensor: ...
def _cast_Int(input: Tensor, non_blocking: _bool = False) -> Tensor: ...
def _cast_Long(input: Tensor, non_blocking: _bool = False) -> Tensor: ...
def _cast_Short(input: Tensor, non_blocking: _bool = False) -> Tensor: ...
def _choose_qparams_per_tensor(
    input: Tensor,
    reduce_range: _bool = False,
) -> tuple[_float, _int]: ...
def _chunk_cat(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    dim: _int,
    num_chunks: _int,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _coalesce(input: Tensor) -> Tensor: ...
def _compute_linear_combination(
    input: Tensor,
    coefficients: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _conj(input: Tensor) -> Tensor: ...
def _conj_copy(input: Tensor, *, out: Tensor | None = None) -> Tensor: ...
def _conj_physical(input: Tensor) -> Tensor: ...
def _convert_indices_from_coo_to_csr(
    input: Tensor,
    size: _int,
    *,
    out_int32: _bool = False,
    out: Tensor | None = None,
) -> Tensor: ...
def _convert_indices_from_csr_to_coo(
    crow_indices: Tensor,
    col_indices: Tensor,
    *,
    out_int32: _bool = False,
    transpose: _bool = False,
    out: Tensor | None = None,
) -> Tensor: ...
def _convert_weight_to_int4pack(input: Tensor, innerKTiles: _int) -> Tensor: ...
def _convert_weight_to_int4pack_for_cpu(
    input: Tensor,
    innerKTiles: _int,
) -> Tensor: ...
@overload
def _convolution(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Sequence[_int | SymInt],
    padding: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    transposed: _bool,
    output_padding: _size,
    groups: _int | SymInt,
    benchmark: _bool,
    deterministic: _bool,
    cudnn_enabled: _bool,
) -> Tensor: ...
@overload
def _convolution(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Sequence[_int | SymInt],
    padding: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    transposed: _bool,
    output_padding: Sequence[_int | SymInt],
    groups: _int | SymInt,
    benchmark: _bool,
    deterministic: _bool,
    cudnn_enabled: _bool,
    allow_tf32: _bool,
) -> Tensor: ...
def _convolution_mode(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Sequence[_int | SymInt],
    padding: str,
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
) -> Tensor: ...
def _copy_from(
    input: Tensor,
    dst: Tensor,
    non_blocking: _bool = False,
) -> Tensor: ...
def _copy_from_and_resize(input: Tensor, dst: Tensor) -> Tensor: ...
def _cslt_compress(input: Tensor) -> Tensor: ...
def _cslt_sparse_mm(
    compressed_A: Tensor,
    dense_B: Tensor,
    bias: Tensor | None = None,
    alpha: Tensor | None = None,
    out_dtype: _dtype | None = None,
    transpose_result: _bool = False,
    alg_id: _int = 0,
    split_k: _int = 1,
    split_k_mode: _int = -1,
) -> Tensor: ...
def _cslt_sparse_mm_search(
    compressed_A: Tensor,
    dense_B: Tensor,
    bias: Tensor | None = None,
    alpha: Tensor | None = None,
    out_dtype: _dtype | None = None,
    transpose_result: _bool = False,
) -> _int: ...
@overload
def _ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: _size,
    target_lengths: _size,
    blank: _int = 0,
    zero_infinity: _bool = False,
) -> tuple[Tensor, Tensor]: ...
@overload
def _ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: _int = 0,
    zero_infinity: _bool = False,
) -> tuple[Tensor, Tensor]: ...
@overload
def _cudnn_ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: _size,
    target_lengths: _size,
    blank: _int,
    deterministic: _bool,
    zero_infinity: _bool,
) -> tuple[Tensor, Tensor]: ...
@overload
def _cudnn_ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: _int,
    deterministic: _bool,
    zero_infinity: _bool,
) -> tuple[Tensor, Tensor]: ...
def _cudnn_init_dropout_state(
    dropout: _float,
    train: _bool,
    dropout_seed: _int,
    *,
    dtype: _dtype | None = None,
    layout: _layout | None = None,
    device: DeviceLikeType | None = None,
    pin_memory: _bool | None = False,
    requires_grad: _bool | None = False,
) -> Tensor: ...
def _cudnn_rnn(
    input: Tensor,
    weight: tuple[Tensor, ...] | list[Tensor] | None,
    weight_stride0: _int,
    weight_buf: Tensor | None,
    hx: Tensor,
    cx: Tensor | None,
    mode: _int,
    hidden_size: _int | SymInt,
    proj_size: _int | SymInt,
    num_layers: _int,
    batch_first: _bool,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    batch_sizes: Sequence[_int | SymInt],
    dropout_state: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
def _cudnn_rnn_flatten_weight(
    weight_arr: tuple[Tensor, ...] | list[Tensor] | None,
    weight_stride0: _int,
    input_size: _int | SymInt,
    mode: _int,
    hidden_size: _int | SymInt,
    proj_size: _int | SymInt,
    num_layers: _int,
    batch_first: _bool,
    bidirectional: _bool,
) -> Tensor: ...
def _cufft_clear_plan_cache(device_index: _int) -> None: ...
def _cufft_get_plan_cache_max_size(device_index: _int) -> _int: ...
def _cufft_get_plan_cache_size(device_index: _int) -> _int: ...
def _cufft_set_plan_cache_max_size(
    device_index: _int,
    max_size: _int,
) -> None: ...
def _cummax_helper(
    input: Tensor,
    values: Tensor,
    indices: Tensor,
    dim: _int,
) -> None: ...
def _cummin_helper(
    input: Tensor,
    values: Tensor,
    indices: Tensor,
    dim: _int,
) -> None: ...
def _debug_has_internal_overlap(input: Tensor) -> _int: ...
def _dim_arange(like: Tensor, dim: _int) -> Tensor: ...
def _dirichlet_grad(x: Tensor, alpha: Tensor, total: Tensor) -> Tensor: ...
def _disable_functionalization(): ...
def _dyn_quant_matmul_4bit(
    inp: Tensor,
    packed_weights: Tensor,
    block_size: _int,
    in_features: _int,
    out_features: _int,
) -> Tensor: ...
def _dyn_quant_pack_4bit_weight(
    weights: Tensor,
    scales_zeros: Tensor,
    bias: Tensor | None,
    block_size: _int,
    in_features: _int,
    out_features: _int,
) -> Tensor: ...
@overload
def _efficientzerotensor(
    size: Sequence[_int | SymInt],
    *,
    dtype: _dtype | None = None,
    layout: _layout | None = None,
    device: DeviceLikeType | None = None,
    pin_memory: _bool | None = False,
    requires_grad: _bool | None = False,
) -> Tensor: ...
@overload
def _efficientzerotensor(
    *size: _int | SymInt,
    dtype: _dtype | None = None,
    layout: _layout | None = None,
    device: DeviceLikeType | None = None,
    pin_memory: _bool | None = False,
    requires_grad: _bool | None = False,
) -> Tensor: ...
def _embedding_bag(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: _bool = False,
    mode: _int = 0,
    sparse: _bool = False,
    per_sample_weights: Tensor | None = None,
    include_last_offset: _bool = False,
    padding_idx: _int = -1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
def _embedding_bag_forward_only(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: _bool = False,
    mode: _int = 0,
    sparse: _bool = False,
    per_sample_weights: Tensor | None = None,
    include_last_offset: _bool = False,
    padding_idx: _int = -1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
@overload
def _empty_affine_quantized(
    size: Sequence[_int | SymInt],
    *,
    scale: _float = 1,
    zero_point: _int = 0,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = None,
    layout: _layout | None = None,
    device: DeviceLikeType | None = None,
    pin_memory: _bool | None = False,
    requires_grad: _bool | None = False,
) -> Tensor: ...
@overload
def _empty_affine_quantized(
    *size: _int | SymInt,
    scale: _float = 1,
    zero_point: _int = 0,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = None,
    layout: _layout | None = None,
    device: DeviceLikeType | None = None,
    pin_memory: _bool | None = False,
    requires_grad: _bool | None = False,
) -> Tensor: ...
@overload
def _empty_per_channel_affine_quantized(
    size: Sequence[_int | SymInt],
    *,
    scales: Tensor,
    zero_points: Tensor,
    axis: _int,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = None,
    layout: _layout | None = None,
    device: DeviceLikeType | None = None,
    pin_memory: _bool | None = False,
    requires_grad: _bool | None = False,
) -> Tensor: ...
@overload
def _empty_per_channel_affine_quantized(
    *size: _int | SymInt,
    scales: Tensor,
    zero_points: Tensor,
    axis: _int,
    memory_format: memory_format | None = ...,
    dtype: _dtype | None = None,
    layout: _layout | None = None,
    device: DeviceLikeType | None = None,
    pin_memory: _bool | None = False,
    requires_grad: _bool | None = False,
) -> Tensor: ...
def _enable_functionalization(*, reapply_views: _bool = False) -> None: ...
def _euclidean_dist(x1: Tensor, x2: Tensor) -> Tensor: ...
def _fake_quantize_learnable_per_channel_affine(
    input: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    axis: _int,
    quant_min: _int,
    quant_max: _int,
    grad_factor: _float = 1.0,
) -> Tensor: ...
def _fake_quantize_learnable_per_tensor_affine(
    input: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    quant_min: _int,
    quant_max: _int,
    grad_factor: _float = 1.0,
) -> Tensor: ...
def _fft_c2c(
    input: Tensor,
    dim: Sequence[_int | SymInt],
    normalization: _int,
    forward: _bool,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _fft_c2r(
    input: Tensor,
    dim: _size,
    normalization: _int,
    last_dim_size: _int | SymInt,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _fft_r2c(
    input: Tensor,
    dim: _size,
    normalization: _int,
    onesided: _bool,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _fill_mem_eff_dropout_mask_(
    input: Tensor,
    dropout_p: _float,
    seed: _int,
    offset: _int,
) -> Tensor: ...
def _foobar(
    input: Tensor,
    arg1: _bool = True,
    arg2: _bool = True,
    *,
    arg3: _bool = True,
) -> Tensor: ...
def _foreach_abs(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_abs_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_acos(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_acos_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
@overload
def _foreach_add(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_add(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    alpha: Number | _complex = 1,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_add(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: Tensor,
    *,
    alpha: Number | _complex = 1,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_add(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_add_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_add_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    alpha: Number | _complex = 1,
) -> None: ...
@overload
def _foreach_add_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: Tensor,
    *,
    alpha: Number | _complex = 1,
) -> None: ...
@overload
def _foreach_add_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> None: ...
@overload
def _foreach_addcdiv(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_addcdiv(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Tensor,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_addcdiv(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    value: Number | _complex = 1,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_addcdiv_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_addcdiv_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Tensor,
) -> None: ...
@overload
def _foreach_addcdiv_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    value: Number | _complex = 1,
) -> None: ...
@overload
def _foreach_addcmul(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_addcmul(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Tensor,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_addcmul(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    value: Number | _complex = 1,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_addcmul_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_addcmul_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Tensor,
) -> None: ...
@overload
def _foreach_addcmul_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensor1: tuple[Tensor, ...] | list[Tensor] | None,
    tensor2: tuple[Tensor, ...] | list[Tensor] | None,
    value: Number | _complex = 1,
) -> None: ...
def _foreach_asin(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_asin_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_atan(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_atan_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_ceil(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_ceil_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
@overload
def _foreach_clamp_max(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_clamp_max(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_clamp_max(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_clamp_max_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_clamp_max_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> None: ...
@overload
def _foreach_clamp_max_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
@overload
def _foreach_clamp_min(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_clamp_min(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_clamp_min(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_clamp_min_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_clamp_min_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> None: ...
@overload
def _foreach_clamp_min_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
def _foreach_cos(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_cos_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_cosh(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_cosh_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
@overload
def _foreach_div(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_div(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: Tensor,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_div(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_div(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_div_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_div_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: Tensor,
) -> None: ...
@overload
def _foreach_div_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> None: ...
@overload
def _foreach_div_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
def _foreach_erf(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_erf_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_erfc(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_erfc_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_exp(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_exp_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_expm1(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_expm1_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_floor(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_floor_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_frac(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_frac_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
@overload
def _foreach_lerp(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensors1: tuple[Tensor, ...] | list[Tensor] | None,
    weight: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_lerp(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensors1: tuple[Tensor, ...] | list[Tensor] | None,
    weight: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_lerp(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensors1: tuple[Tensor, ...] | list[Tensor] | None,
    weights: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_lerp_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensors1: tuple[Tensor, ...] | list[Tensor] | None,
    weight: Number | _complex,
) -> None: ...
@overload
def _foreach_lerp_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensors1: tuple[Tensor, ...] | list[Tensor] | None,
    weight: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_lerp_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    tensors1: tuple[Tensor, ...] | list[Tensor] | None,
    weights: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
def _foreach_lgamma(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_lgamma_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
def _foreach_log(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_log1p(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_log1p_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_log2(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_log2_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_log10(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_log10_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_log_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_max(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_maximum(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_maximum(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_maximum(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_maximum_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_maximum_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> None: ...
@overload
def _foreach_maximum_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
@overload
def _foreach_minimum(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_minimum(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_minimum(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_minimum_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_minimum_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> None: ...
@overload
def _foreach_minimum_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
@overload
def _foreach_mul(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_mul(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: Tensor,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_mul(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_mul(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_mul_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_mul_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: Tensor,
) -> None: ...
@overload
def _foreach_mul_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> None: ...
@overload
def _foreach_mul_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
def _foreach_neg(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_neg_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_norm(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    ord: Number | _complex = 2,
    dtype: _dtype | None = None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_pow(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    exponent: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_pow(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    exponent: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_pow(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    exponent: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_pow(
    self: Number | _complex,
    exponent: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_pow_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    exponent: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_pow_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    exponent: Number | _complex,
) -> None: ...
@overload
def _foreach_pow_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    exponent: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
def _foreach_reciprocal(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_reciprocal_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
def _foreach_round(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_round_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_rsqrt(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_rsqrt_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_sigmoid(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_sigmoid_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> None: ...
def _foreach_sign(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_sign_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_sin(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_sin_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_sinh(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_sinh_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_sqrt(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_sqrt_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
@overload
def _foreach_sub(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_sub(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    alpha: Number | _complex = 1,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_sub(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> tuple[Tensor, ...]: ...
@overload
def _foreach_sub_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalars: Sequence[Number | _complex],
) -> None: ...
@overload
def _foreach_sub_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    other: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    alpha: Number | _complex = 1,
) -> None: ...
@overload
def _foreach_sub_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    scalar: Number | _complex,
) -> None: ...
def _foreach_tan(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_tan_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_tanh(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_tanh_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_trunc(
    self: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _foreach_trunc_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _foreach_zero_(self: tuple[Tensor, ...] | list[Tensor] | None) -> None: ...
def _from_functional_tensor(t: Tensor) -> Tensor: ...
def _functional_assert_async(
    input: Tensor,
    assert_msg: str,
    dep_token: Tensor,
) -> Tensor: ...
def _functional_assert_scalar(
    self: Number | _complex,
    assert_msg: str,
    dep_token: Tensor,
) -> Tensor: ...
def _functional_sym_constrain_range(
    size: Number | _complex,
    min: _int | None,
    max: _int | None,
    dep_token: Tensor,
) -> Tensor: ...
def _functional_sym_constrain_range_for_size(
    size: Number | _complex,
    min: _int | None,
    max: _int | None,
    dep_token: Tensor,
) -> Tensor: ...
def _functionalize_apply_view_metas(tensor: Tensor, base: Tensor) -> Tensor: ...
def _functionalize_commit_update(t: Tensor) -> None: ...
def _functionalize_has_metadata_mutation(tensor: Tensor) -> _bool: ...
def _functionalize_inductor_storage_resized_counter(t: Tensor) -> _int: ...
def _functionalize_is_symbolic(tensor: Tensor) -> _bool: ...
def _functionalize_mark_mutation_hidden_from_autograd(t: Tensor) -> None: ...
def _functionalize_mark_storage_changed(tensor: Tensor) -> _bool: ...
def _functionalize_mutation_counter(t: Tensor) -> _int: ...
def _functionalize_replace(self_: Tensor, other: Tensor) -> None: ...
def _functionalize_storage_changed_counter(t: Tensor) -> _int: ...
def _functionalize_sync(t: Tensor) -> None: ...
def _functionalize_unsafe_set(dst: Tensor, src: Tensor) -> None: ...
def _functionalize_was_inductor_storage_resized(t: Tensor) -> _bool: ...
def _functionalize_was_storage_changed(tensor: Tensor) -> _bool: ...
@overload
def _fused_adagrad_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    grads: tuple[Tensor, ...] | list[Tensor] | None,
    state_sums: tuple[Tensor, ...] | list[Tensor] | None,
    state_steps: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    lr: Tensor,
    lr_decay: _float,
    weight_decay: _float,
    eps: _float,
    maximize: _bool,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
) -> None: ...
@overload
def _fused_adagrad_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    grads: tuple[Tensor, ...] | list[Tensor] | None,
    state_sums: tuple[Tensor, ...] | list[Tensor] | None,
    state_steps: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    lr: _float,
    lr_decay: _float,
    weight_decay: _float,
    eps: _float,
    maximize: _bool,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
) -> None: ...
@overload
def _fused_adam_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    grads: tuple[Tensor, ...] | list[Tensor] | None,
    exp_avgs: tuple[Tensor, ...] | list[Tensor] | None,
    exp_avg_sqs: tuple[Tensor, ...] | list[Tensor] | None,
    max_exp_avg_sqs: tuple[Tensor, ...] | list[Tensor] | None,
    state_steps: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    lr: Tensor,
    beta1: _float,
    beta2: _float,
    weight_decay: _float,
    eps: _float,
    amsgrad: _bool,
    maximize: _bool,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
) -> None: ...
@overload
def _fused_adam_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    grads: tuple[Tensor, ...] | list[Tensor] | None,
    exp_avgs: tuple[Tensor, ...] | list[Tensor] | None,
    exp_avg_sqs: tuple[Tensor, ...] | list[Tensor] | None,
    max_exp_avg_sqs: tuple[Tensor, ...] | list[Tensor] | None,
    state_steps: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    lr: _float,
    beta1: _float,
    beta2: _float,
    weight_decay: _float,
    eps: _float,
    amsgrad: _bool,
    maximize: _bool,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
) -> None: ...
@overload
def _fused_adamw_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    grads: tuple[Tensor, ...] | list[Tensor] | None,
    exp_avgs: tuple[Tensor, ...] | list[Tensor] | None,
    exp_avg_sqs: tuple[Tensor, ...] | list[Tensor] | None,
    max_exp_avg_sqs: tuple[Tensor, ...] | list[Tensor] | None,
    state_steps: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    lr: Tensor,
    beta1: _float,
    beta2: _float,
    weight_decay: _float,
    eps: _float,
    amsgrad: _bool,
    maximize: _bool,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
) -> None: ...
@overload
def _fused_adamw_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    grads: tuple[Tensor, ...] | list[Tensor] | None,
    exp_avgs: tuple[Tensor, ...] | list[Tensor] | None,
    exp_avg_sqs: tuple[Tensor, ...] | list[Tensor] | None,
    max_exp_avg_sqs: tuple[Tensor, ...] | list[Tensor] | None,
    state_steps: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    lr: _float,
    beta1: _float,
    beta2: _float,
    weight_decay: _float,
    eps: _float,
    amsgrad: _bool,
    maximize: _bool,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
) -> None: ...
def _fused_dropout(
    input: Tensor,
    p: _float,
    generator: Generator | None = None,
) -> tuple[Tensor, Tensor]: ...
def _fused_moving_avg_obs_fq_helper(
    input: Tensor,
    observer_on: Tensor,
    fake_quant_on: Tensor,
    running_min: Tensor,
    running_max: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    averaging_const: _float,
    quant_min: _int,
    quant_max: _int,
    ch_axis: _int,
    per_row_fake_quant: _bool = False,
    symmetric_quant: _bool = False,
) -> torch.return_types._fused_moving_avg_obs_fq_helper: ...
def _fused_rms_norm(
    input: Tensor,
    normalized_shape: _size,
    weight: Tensor | None,
    eps: _float | None,
) -> tuple[Tensor, Tensor]: ...
def _fused_sdp_choice(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: _float = 0.0,
    is_causal: _bool = False,
    *,
    scale: _float | None = None,
    enable_gqa: _bool = False,
) -> _int: ...
@overload
def _fused_sgd_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    grads: tuple[Tensor, ...] | list[Tensor] | None,
    momentum_buffer_list: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    weight_decay: _float,
    momentum: _float,
    lr: Tensor,
    dampening: _float,
    nesterov: _bool,
    maximize: _bool,
    is_first_step: _bool,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
) -> None: ...
@overload
def _fused_sgd_(
    self: tuple[Tensor, ...] | list[Tensor] | None,
    grads: tuple[Tensor, ...] | list[Tensor] | None,
    momentum_buffer_list: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    weight_decay: _float,
    momentum: _float,
    lr: _float,
    dampening: _float,
    nesterov: _bool,
    maximize: _bool,
    is_first_step: _bool,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
) -> None: ...
def _fw_primal_copy(
    input: Tensor,
    level: _int,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _grid_sampler_2d_cpu_fallback(
    input: Tensor,
    grid: Tensor,
    interpolation_mode: _int,
    padding_mode: _int,
    align_corners: _bool,
) -> Tensor: ...
def _grouped_mm(
    input: Tensor,
    mat2: Tensor,
    offs: Tensor | None = None,
    bias: Tensor | None = None,
    out_dtype: _dtype | None = None,
) -> Tensor: ...
def _has_compatible_shallow_copy_type(
    input: Tensor,
    from_: Tensor,
) -> _bool: ...
def _histogramdd_bin_edges(
    input: Tensor,
    bins: _size,
    *,
    range: Sequence[_float] | None = None,
    weight: Tensor | None = None,
    density: _bool = False,
) -> tuple[Tensor, ...]: ...
def _histogramdd_from_bin_cts(
    input: Tensor,
    bins: _size,
    *,
    range: Sequence[_float] | None = None,
    weight: Tensor | None = None,
    density: _bool = False,
) -> Tensor: ...
def _histogramdd_from_bin_tensors(
    input: Tensor,
    bins: tuple[Tensor, ...] | list[Tensor] | None,
    *,
    weight: Tensor | None = None,
    density: _bool = False,
) -> Tensor: ...
def _index_put_impl_(
    input: Tensor,
    indices: tuple[Tensor, ...] | list[Tensor] | None,
    values: Tensor,
    accumulate: _bool = False,
    unsafe: _bool = False,
) -> Tensor: ...
def _indices_copy(input: Tensor, *, out: Tensor | None = None) -> Tensor: ...
def _int_mm(
    input: Tensor,
    mat2: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _is_all_true(input: Tensor) -> Tensor: ...
def _is_any_true(input: Tensor) -> Tensor: ...
def _is_functional_tensor(t: Tensor) -> _bool: ...
def _is_functional_tensor_base(t: Tensor) -> _bool: ...
def _is_zerotensor(input: Tensor) -> _bool: ...
def _lazy_clone(input: Tensor) -> Tensor: ...
def _linalg_check_errors(
    info: Tensor,
    api_name: str,
    *,
    is_matrix: _bool,
) -> None: ...
def _linalg_det(
    A: Tensor,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = None,
) -> torch.return_types._linalg_det: ...
def _linalg_eigh(
    A: Tensor,
    UPLO: str = "L",
    compute_v: _bool = True,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = None,
) -> torch.return_types._linalg_eigh: ...
def _linalg_slogdet(
    A: Tensor,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = None,
) -> torch.return_types._linalg_slogdet: ...
def _linalg_solve_ex(
    A: Tensor,
    B: Tensor,
    *,
    left: _bool = True,
    check_errors: _bool = False,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = None,
) -> torch.return_types._linalg_solve_ex: ...
def _linalg_svd(
    A: Tensor,
    full_matrices: _bool = False,
    compute_uv: _bool = True,
    *,
    driver: str | None = None,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = None,
) -> torch.return_types._linalg_svd: ...
def _log_softmax(
    input: Tensor,
    dim: _int,
    half_to_float: _bool,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _log_softmax_backward_data(
    grad_output: Tensor,
    output: Tensor,
    dim: _int,
    input_dtype: _dtype,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _logcumsumexp(
    input: Tensor,
    dim: _int,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _lstm_mps(
    input: Tensor,
    hx: tuple[Tensor, ...] | list[Tensor] | None,
    params: tuple[Tensor, ...] | list[Tensor] | None,
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    batch_first: _bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...
def _lu_with_info(
    input: Tensor,
    pivot: _bool = True,
    check_errors: _bool = True,
) -> torch.return_types._lu_with_info: ...
def _make_dep_token(
    *,
    memory_format: memory_format | None = None,
    dtype: _dtype | None = None,
    layout: _layout | None = None,
    device: DeviceLikeType | None = None,
    pin_memory: _bool | None = False,
    requires_grad: _bool | None = False,
) -> Tensor: ...
def _make_dual(primal: Tensor, tangent: Tensor, level: _int) -> Tensor: ...
def _make_dual_copy(
    primal: Tensor,
    tangent: Tensor,
    level: _int,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _make_per_channel_quantized_tensor(
    input: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    axis: _int,
) -> Tensor: ...
def _make_per_tensor_quantized_tensor(
    input: Tensor,
    scale: _float,
    zero_point: _int,
) -> Tensor: ...
def _masked_scale(input: Tensor, mask: Tensor, scale: _float) -> Tensor: ...
def _masked_softmax(
    input: Tensor,
    mask: Tensor,
    dim: _int | None = None,
    mask_type: _int | None = None,
) -> Tensor: ...
def _mixed_dtypes_linear(
    input: Tensor,
    weight: Tensor,
    scale: Tensor,
    *,
    bias: Tensor | None = None,
    activation: str | None = None,
) -> Tensor: ...
def _mkldnn_reshape(input: Tensor, shape: _size) -> Tensor: ...
def _mkldnn_transpose(input: Tensor, dim0: _int, dim1: _int) -> Tensor: ...
def _mkldnn_transpose_(input: Tensor, dim0: _int, dim1: _int) -> Tensor: ...
def _mps_convolution(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    padding: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
) -> Tensor: ...
def _mps_convolution_transpose(
    input: Tensor,
    weight: Tensor,
    padding: Sequence[_int | SymInt],
    output_padding: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    dilation: Sequence[_int | SymInt],
    groups: _int | SymInt,
) -> Tensor: ...
@overload
def _native_batch_norm_legit(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor,
    running_var: Tensor,
    training: _bool,
    momentum: _float,
    eps: _float,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]: ...
@overload
def _native_batch_norm_legit(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    training: _bool,
    momentum: _float,
    eps: _float,
    *,
    out: Tensor | tuple[Tensor, ...] | list[Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]: ...
def _native_batch_norm_legit_no_training(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: _float,
    eps: _float,
) -> tuple[Tensor, Tensor, Tensor]: ...
def _native_multi_head_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim: _int,
    num_head: _int,
    qkv_weight: Tensor,
    qkv_bias: Tensor,
    proj_weight: Tensor,
    proj_bias: Tensor,
    mask: Tensor | None = None,
    need_weights: _bool = True,
    average_attn_weights: _bool = True,
    mask_type: _int | None = None,
) -> tuple[Tensor, Tensor]: ...
def _neg_view(input: Tensor) -> Tensor: ...
def _neg_view_copy(input: Tensor, *, out: Tensor | None = None) -> Tensor: ...
def _nested_compute_contiguous_strides_offsets(
    nested_size: Tensor,
) -> tuple[Tensor, Tensor]: ...
def _nested_from_padded(
    padded: Tensor,
    cpu_nested_shape_example: Tensor,
    fuse_transform_0213: _bool = False,
) -> Tensor: ...
def _nested_from_padded_and_nested_example(
    padded: Tensor,
    nt_example: Tensor,
) -> Tensor: ...
def _nested_from_padded_tensor(
    padded: Tensor,
    offsets: Tensor,
    dummy: Tensor,
    ragged_idx: _int = 1,
    min_seqlen: Tensor | None = None,
    max_seqlen: Tensor | None = None,
    sum_S: _int | SymInt | None = None,
) -> Tensor: ...
def _nested_get_jagged_dummy(any: Tensor) -> Tensor: ...
def _nested_get_lengths(input: Tensor) -> Tensor: ...
def _nested_get_max_seqlen(input: Tensor) -> Tensor: ...
def _nested_get_min_seqlen(input: Tensor) -> Tensor: ...
def _nested_get_offsets(input: Tensor) -> Tensor: ...
def _nested_get_ragged_idx(input: Tensor) -> _int: ...
def _nested_get_values(input: Tensor) -> Tensor: ...
def _nested_get_values_copy(
    input: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _nested_tensor_from_mask(
    t: Tensor,
    mask: Tensor,
    mask_check: _bool = True,
) -> Tensor: ...
def _nested_tensor_from_mask_left_aligned(t: Tensor, mask: Tensor) -> _bool: ...
def _nested_tensor_from_tensor_list(
    list: tuple[Tensor, ...] | list[Tensor] | None,
    dtype: _dtype | None = None,
    layout: _layout | None = None,
    device: DeviceLikeType | None = None,
    pin_memory: _bool | None = None,
) -> Tensor: ...
def _nested_tensor_softmax_with_shape(
    input: Tensor,
    query: Tensor,
) -> Tensor: ...
def _nested_view_from_buffer(
    input: Tensor,
    nested_size: Tensor,
    nested_strides: Tensor,
    offsets: Tensor,
) -> Tensor: ...
def _nested_view_from_buffer_copy(
    input: Tensor,
    nested_size: Tensor,
    nested_strides: Tensor,
    offsets: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _nested_view_from_jagged(
    input: Tensor,
    offsets: Tensor,
    dummy: Tensor,
    lengths: Tensor | None = None,
    ragged_idx: _int = 1,
    min_seqlen: Tensor | None = None,
    max_seqlen: Tensor | None = None,
) -> Tensor: ...
def _nested_view_from_jagged_copy(
    input: Tensor,
    offsets: Tensor,
    dummy: Tensor,
    lengths: Tensor | None = None,
    ragged_idx: _int = 1,
    min_seqlen: Tensor | None = None,
    max_seqlen: Tensor | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _nnpack_available() -> _bool: ...
def _nnpack_spatial_convolution(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    padding: _int | SymInt | Sequence[_int | SymInt],
    stride: _int | SymInt | Sequence[_int | SymInt] = 1,
) -> Tensor: ...
def _pack_padded_sequence(
    input: Tensor,
    lengths: Tensor,
    batch_first: _bool,
) -> tuple[Tensor, Tensor]: ...
def _pad_packed_sequence(
    data: Tensor,
    batch_sizes: Tensor,
    batch_first: _bool,
    padding_value: Number | _complex,
    total_length: _int,
) -> tuple[Tensor, Tensor]: ...
def _pin_memory(
    input: Tensor,
    device: DeviceLikeType | None = None,
) -> Tensor: ...
def _prelu_kernel(input: Tensor, weight: Tensor) -> Tensor: ...
def _print(s: str) -> None: ...
def _propagate_xla_data(input: Tensor, output: Tensor) -> None: ...
def _remove_batch_dim(
    input: Tensor,
    level: _int,
    batch_size: _int | SymInt,
    out_dim: _int,
) -> Tensor: ...
def _reshape_alias_copy(
    input: Tensor,
    size: Sequence[_int | SymInt],
    stride: Sequence[_int | SymInt],
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _reshape_from_tensor(input: Tensor, shape: Tensor) -> Tensor: ...
def _resize_output_(
    input: Tensor,
    size: Sequence[_int | SymInt],
    device: DeviceLikeType | None,
) -> Tensor: ...
def _rowwise_prune(
    weight: Tensor,
    mask: Tensor,
    compressed_indices_dtype: _dtype,
) -> tuple[Tensor, Tensor]: ...
def _safe_softmax(
    input: Tensor,
    dim: _int,
    dtype: _dtype | None = None,
) -> Tensor: ...
def _sample_dirichlet(
    input: Tensor,
    generator: Generator | None = None,
) -> Tensor: ...
def _saturate_weight_to_fp16(weight: Tensor) -> Tensor: ...
def _scaled_dot_product_attention_math(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: _float = 0.0,
    is_causal: _bool = False,
    dropout_mask: Tensor | None = None,
    *,
    scale: _float | None = None,
    enable_gqa: _bool = False,
) -> tuple[Tensor, Tensor]: ...
def _scaled_dot_product_attention_math_for_mps(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: _float = 0.0,
    is_causal: _bool = False,
    dropout_mask: Tensor | None = None,
    *,
    scale: _float | None = None,
) -> tuple[Tensor, Tensor]: ...
def _scaled_dot_product_cudnn_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Tensor | None,
    compute_log_sumexp: _bool,
    dropout_p: _float = 0.0,
    is_causal: _bool = False,
    return_debug_mask: _bool = False,
    *,
    scale: _float | None = None,
) -> torch.return_types._scaled_dot_product_cudnn_attention: ...
def _scaled_dot_product_efficient_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Tensor | None,
    compute_log_sumexp: _bool,
    dropout_p: _float = 0.0,
    is_causal: _bool = False,
    *,
    scale: _float | None = None,
) -> torch.return_types._scaled_dot_product_efficient_attention: ...
@overload
def _scaled_dot_product_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_descale: Tensor | None,
    k_descale: Tensor | None,
    v_descale: Tensor | None,
    dropout_p: _float = 0.0,
    is_causal: _bool = False,
    return_debug_mask: _bool = False,
    *,
    scale: _float | None = None,
) -> torch.return_types._scaled_dot_product_flash_attention: ...
@overload
def _scaled_dot_product_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: _float = 0.0,
    is_causal: _bool = False,
    return_debug_mask: _bool = False,
    *,
    scale: _float | None = None,
) -> torch.return_types._scaled_dot_product_flash_attention: ...
def _scaled_dot_product_flash_attention_for_cpu(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: _float = 0.0,
    is_causal: _bool = False,
    *,
    attn_mask: Tensor | None = None,
    scale: _float | None = None,
) -> torch.return_types._scaled_dot_product_flash_attention_for_cpu: ...
def _scaled_grouped_mm(
    input: Tensor,
    mat2: Tensor,
    scale_a: Tensor,
    scale_b: Tensor,
    offs: Tensor | None = None,
    bias: Tensor | None = None,
    scale_result: Tensor | None = None,
    out_dtype: _dtype | None = None,
    use_fast_accum: _bool = False,
) -> Tensor: ...
def _scaled_mm(
    input: Tensor,
    mat2: Tensor,
    scale_a: Tensor,
    scale_b: Tensor,
    bias: Tensor | None = None,
    scale_result: Tensor | None = None,
    out_dtype: _dtype | None = None,
    use_fast_accum: _bool = False,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _shape_as_tensor(input: Tensor) -> Tensor: ...
def _sobol_engine_draw(
    quasi: Tensor,
    n: _int,
    sobolstate: Tensor,
    dimension: _int,
    num_generated: _int,
    dtype: _dtype | None,
) -> tuple[Tensor, Tensor]: ...
def _sobol_engine_ff_(
    input: Tensor,
    n: _int,
    sobolstate: Tensor,
    dimension: _int,
    num_generated: _int,
) -> Tensor: ...
def _sobol_engine_initialize_state_(
    input: Tensor,
    dimension: _int,
) -> Tensor: ...
def _sobol_engine_scramble_(
    input: Tensor,
    ltm: Tensor,
    dimension: _int,
) -> Tensor: ...
def _softmax(
    input: Tensor,
    dim: _int,
    half_to_float: _bool,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _softmax_backward_data(
    grad_output: Tensor,
    output: Tensor,
    dim: _int,
    input_dtype: _dtype,
    *,
    grad_input: Tensor | None = None,
) -> Tensor: ...
def _sparse_broadcast_to(input: Tensor, size: _size) -> Tensor: ...
def _sparse_broadcast_to_copy(
    input: Tensor,
    size: _size,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _sparse_csr_prod(
    input: Tensor,
    dim: _int | _size,
    keepdim: _bool = False,
    *,
    dtype: _dtype | None = None,
) -> Tensor: ...
def _sparse_csr_sum(
    input: Tensor,
    dim: _int | _size,
    keepdim: _bool = False,
    *,
    dtype: _dtype | None = None,
) -> Tensor: ...
def _sparse_log_softmax_backward_data(
    grad_output: Tensor,
    output: Tensor,
    dim: _int,
    input: Tensor,
) -> Tensor: ...
def _sparse_semi_structured_addmm(
    input: Tensor,
    mat1: Tensor,
    mat1_meta: Tensor,
    mat2: Tensor,
    *,
    alpha: Number | _complex = 1,
    beta: Number | _complex = 1,
    out_dtype: _dtype | None = None,
) -> Tensor: ...
def _sparse_semi_structured_apply(
    input: Tensor,
    thread_masks: Tensor,
) -> tuple[Tensor, Tensor]: ...
def _sparse_semi_structured_apply_dense(
    input: Tensor,
    thread_masks: Tensor,
) -> Tensor: ...
def _sparse_semi_structured_linear(
    input: Tensor,
    weight: Tensor,
    meta: Tensor,
    *,
    bias: Tensor | None = None,
    activation: str | None = None,
    out_dtype: _dtype | None = None,
) -> Tensor: ...
def _sparse_semi_structured_mm(
    mat1: Tensor,
    mat1_meta: Tensor,
    mat2: Tensor,
    *,
    out_dtype: _dtype | None = None,
) -> Tensor: ...
def _sparse_semi_structured_tile(
    input: Tensor,
    algorithm: str = "",
    use_cutlass: _bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
def _sparse_softmax_backward_data(
    grad_output: Tensor,
    output: Tensor,
    dim: _int,
    input: Tensor,
) -> Tensor: ...
def _sparse_sparse_matmul(input: Tensor, other: Tensor) -> Tensor: ...
@overload
def _sparse_sum(input: Tensor) -> Tensor: ...
@overload
def _sparse_sum(input: Tensor, *, dtype: _dtype) -> Tensor: ...
@overload
def _sparse_sum(input: Tensor, dim: _int | _size) -> Tensor: ...
@overload
def _sparse_sum(
    input: Tensor,
    dim: _int | _size,
    *,
    dtype: _dtype,
) -> Tensor: ...
def _stack(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    dim: _int = 0,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _standard_gamma(
    input: Tensor,
    generator: Generator | None = None,
) -> Tensor: ...
def _standard_gamma_grad(input: Tensor, output: Tensor) -> Tensor: ...
def _sync(t: Tensor) -> None: ...
@overload
def _test_autograd_multiple_dispatch(input: Tensor) -> Tensor: ...
@overload
def _test_autograd_multiple_dispatch(input: Tensor, b: _bool) -> Tensor: ...
def _test_autograd_multiple_dispatch_view(input: Tensor) -> Tensor: ...
def _test_autograd_multiple_dispatch_view_copy(
    input: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def _test_check_tensor(input: Tensor) -> Tensor: ...
def _test_functorch_fallback(input: Tensor, other: Tensor) -> Tensor: ...
def _test_parallel_materialize(
    input: Tensor,
    num_parallel: _int,
    skip_first: _bool = False,
) -> Tensor: ...
def _test_serialization_subcmul(
    input: Tensor,
    other: Tensor,
    alpha: Number | _complex = 1,
) -> Tensor: ...
def _to_cpu(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
) -> tuple[Tensor, ...]: ...
def _to_functional_tensor(t: Tensor) -> Tensor: ...
def _to_sparse_semi_structured(dense: Tensor) -> tuple[Tensor, Tensor]: ...
def _transform_bias_rescale_qkv(
    qkv: Tensor,
    qkv_bias: Tensor,
    num_heads: _int,
) -> tuple[Tensor, Tensor, Tensor]: ...
def _transformer_encoder_layer_fwd(
    src: Tensor,
    embed_dim: _int,
    num_heads: _int,
    qkv_weight: Tensor,
    qkv_bias: Tensor,
    proj_weight: Tensor,
    proj_bias: Tensor,
    use_gelu: _bool,
    norm_first: _bool,
    eps: _float,
    norm_weight_1: Tensor,
    norm_bias_1: Tensor,
    norm_weight_2: Tensor,
    norm_bias_2: Tensor,
    ffn_weight_1: Tensor,
    ffn_bias_1: Tensor,
    ffn_weight_2: Tensor,
    ffn_bias_2: Tensor,
    mask: Tensor | None = None,
    mask_type: _int | None = None,
) -> Tensor: ...
def _trilinear(
    i1: Tensor,
    i2: Tensor,
    i3: Tensor,
    expand1: _size,
    expand2: _size,
    expand3: _size,
    sumdim: _size,
    unroll_dim: _int = 1,
) -> Tensor: ...
def _triton_multi_head_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim: _int,
    num_head: _int,
    qkv_weight: Tensor,
    qkv_bias: Tensor,
    proj_weight: Tensor,
    proj_bias: Tensor,
    mask: Tensor | None = None,
) -> Tensor: ...
def _triton_scaled_dot_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: _float = 0.0,
) -> Tensor: ...
def _unique(
    input: Tensor,
    sorted: _bool = True,
    return_inverse: _bool = False,
) -> tuple[Tensor, Tensor]: ...
def _unique2(
    input: Tensor,
    sorted: _bool = True,
    return_inverse: _bool = False,
    return_counts: _bool = False,
) -> tuple[Tensor, Tensor, Tensor]: ...
def _unpack_dual(
    dual: Tensor,
    level: _int,
) -> torch.return_types._unpack_dual: ...
def _unsafe_index(
    input: Tensor,
    indices: tuple[Tensor, ...] | list[Tensor] | None,
) -> Tensor: ...
def _unsafe_index_put(
    input: Tensor,
    indices: tuple[Tensor, ...] | list[Tensor] | None,
    values: Tensor,
    accumulate: _bool = False,
) -> Tensor: ...
def _unsafe_masked_index(
    input: Tensor,
    mask: Tensor,
    indices: tuple[Tensor, ...] | list[Tensor] | None,
    fill: Number | _complex,
) -> Tensor: ...
def _unsafe_masked_index_put_accumulate(
    input: Tensor,
    mask: Tensor,
    indices: tuple[Tensor, ...] | list[Tensor] | None,
    values: Tensor,
) -> Tensor: ...
@overload
def _use_cudnn_ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: _int,
) -> _bool: ...
@overload
def _use_cudnn_ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: _size,
    target_lengths: _size,
    blank: _int,
) -> _bool: ...
def _use_cudnn_rnn_flatten_weight() -> _bool: ...
def _validate_compressed_sparse_indices(
    is_crow: _bool,
    compressed_idx: Tensor,
    plain_idx: Tensor,
    cdim: _int,
    dim: _int,
    nnz: _int,
) -> None: ...
def _validate_sparse_bsc_tensor_args(
    ccol_indices: Tensor,
    row_indices: Tensor,
    values: Tensor,
    size: _size,
    check_pinning: _bool | None = None,
) -> None: ...
def _validate_sparse_bsr_tensor_args(
    crow_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    size: _size,
    check_pinning: _bool | None = None,
) -> None: ...
def _validate_sparse_compressed_tensor_args(
    compressed_indices: Tensor,
    plain_indices: Tensor,
    values: Tensor,
    size: _size,
    layout: _layout,
    check_pinning: _bool | None = None,
) -> None: ...
def _validate_sparse_coo_tensor_args(
    indices: Tensor,
    values: Tensor,
    size: _size,
    is_coalesced: _bool | None = None,
    check_pinning: _bool | None = None,
) -> None: ...
def _validate_sparse_csc_tensor_args(
    ccol_indices: Tensor,
    row_indices: Tensor,
    values: Tensor,
    size: _size,
    check_pinning: _bool | None = None,
) -> None: ...
def _validate_sparse_csr_tensor_args(
    crow_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    size: _size,
    check_pinning: _bool | None = None,
) -> None: ...
def _values_copy(input: Tensor, *, out: Tensor | None = None) -> Tensor: ...
def _weight_int4pack_mm(
    input: Tensor,
    mat2: Tensor,
    qGroupSize: _int,
    qScaleAndZeros: Tensor,
) -> Tensor: ...
def _weight_int4pack_mm_for_cpu(
    input: Tensor,
    mat2: Tensor,
    qGroupSize: _int,
    qScaleAndZeros: Tensor,
) -> Tensor: ...
def _weight_int4pack_mm_with_scales_and_zeros(
    input: Tensor,
    mat2: Tensor,
    qGroupSize: _int,
    qScale: Tensor,
    qZeros: Tensor,
) -> Tensor: ...
def _weight_int8pack_mm(
    input: Tensor,
    mat2: Tensor,
    scales: Tensor,
) -> Tensor: ...
def _weight_norm(v: Tensor, g: Tensor, dim: _int = 0) -> Tensor: ...
def _weight_norm_interface(
    v: Tensor,
    g: Tensor,
    dim: _int = 0,
) -> tuple[Tensor, Tensor]: ...
def _wrapped_linear_prepack(
    weight: Tensor,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    bias: Tensor,
) -> Tensor: ...
def _wrapped_quantized_linear_prepacked(
    input: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    packed_weight: Tensor,
    output_scale: Tensor,
    output_zero_point: Tensor,
    out_channel: _int,
) -> Tensor: ...
