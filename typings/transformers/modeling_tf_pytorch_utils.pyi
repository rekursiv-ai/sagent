from .utils import ExplicitEnum

"""PyTorch - TF 2.0 general utilities."""
logger = ...

class TransposeType(ExplicitEnum):
    NO = ...
    SIMPLE = ...
    CONV1D = ...
    CONV2D = ...

def convert_tf_weight_name_to_pt_weight_name(
    tf_name, start_prefix_to_remove=..., tf_weight_shape=..., name_scope=...
):  # -> tuple[str, Literal[TransposeType.CONV2D, TransposeType.CONV1D, TransposeType.SIMPLE, TransposeType.NO]]:
    ...
def apply_transpose(
    transpose: TransposeType, weight, match_shape=..., pt_to_tf=...
):  # -> NDArray[Any] | Array:
    ...
def load_pytorch_checkpoint_in_tf2_model(
    tf_model,
    pytorch_checkpoint_path,
    tf_inputs=...,
    allow_missing_keys=...,
    output_loading_info=...,
    _prefix=...,
    tf_to_pt_weight_rename=...,
):  # -> tuple[Any, dict[str, list[Any]]]:
    ...
def load_pytorch_model_in_tf2_model(
    tf_model, pt_model, tf_inputs=..., allow_missing_keys=...
):  # -> tuple[Any, dict[str, list[Any]]]:
    ...
def load_pytorch_weights_in_tf2_model(
    tf_model,
    pt_state_dict,
    tf_inputs=...,
    allow_missing_keys=...,
    output_loading_info=...,
    _prefix=...,
    tf_to_pt_weight_rename=...,
):  # -> tuple[Any, dict[str, list[Any]]]:
    ...
def load_pytorch_state_dict_in_tf2_model(
    tf_model,
    pt_state_dict,
    tf_inputs=...,
    allow_missing_keys=...,
    output_loading_info=...,
    _prefix=...,
    tf_to_pt_weight_rename=...,
    ignore_mismatched_sizes=...,
    skip_logger_warnings=...,
):  # -> tuple[Any, dict[str, list[Any]]]:
    ...
def load_sharded_pytorch_safetensors_in_tf2_model(
    tf_model,
    safetensors_shards,
    tf_inputs=...,
    allow_missing_keys=...,
    output_loading_info=...,
    _prefix=...,
    tf_to_pt_weight_rename=...,
    ignore_mismatched_sizes=...,
):  # -> tuple[Any, dict[str, list[Any] | Any]]:
    ...
def load_tf2_checkpoint_in_pytorch_model(
    pt_model,
    tf_checkpoint_path,
    tf_inputs=...,
    allow_missing_keys=...,
    output_loading_info=...,
):  # -> tuple[Any, dict[str, Any | list[Any]]]:
    ...
def load_tf2_model_in_pytorch_model(
    pt_model, tf_model, allow_missing_keys=..., output_loading_info=...
):  # -> tuple[Any, dict[str, Any | list[Any]]]:
    ...
def load_tf2_weights_in_pytorch_model(
    pt_model, tf_weights, allow_missing_keys=..., output_loading_info=...
):  # -> tuple[Any, dict[str, Any | list[Any]]]:
    ...
def load_tf2_state_dict_in_pytorch_model(
    pt_model, tf_state_dict, allow_missing_keys=..., output_loading_info=...
):  # -> tuple[Any, dict[str, Any | list[Any]]]:
    ...
