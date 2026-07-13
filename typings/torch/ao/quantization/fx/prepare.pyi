from typing import Any

from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node

from .custom_config import PrepareCustomConfig
from .match_utils import _MatchResultWithQConfig

__all__ = ["insert_observers_for_model", "prepare", "propagate_dtypes_for_known_nodes"]
_DO_NOT_OBS_DTYPE_LIST = ...
_OBS_DTYPE_LIST = ...
_DEFAULT_FP32_OBS_OR_FQ_CTR = ...
_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO = ...
_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO = ...

def propagate_dtypes_for_known_nodes(
    graph: Graph,
    node_name_to_match_result_with_qconfig: dict[str, _MatchResultWithQConfig],
) -> None: ...
def insert_observers_for_model(
    model: GraphModule,
    node_name_to_match_result_with_qconfig: dict[str, _MatchResultWithQConfig],
    node_name_to_qconfig: dict[str, QConfigAny],
    prepare_custom_config: PrepareCustomConfig,
    equalization_config_map: dict[str, Any],
    backend_config: BackendConfig,
    observed_node_names: set[str],
    is_qat: bool,
) -> Node | None: ...
def prepare(
    model: GraphModule,
    qconfig_mapping: QConfigMapping | dict[str, Any],
    is_qat: bool,
    node_name_to_scope: dict[str, tuple[str, type]],
    example_inputs: tuple[Any, ...],
    prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = ...,
    _equalization_config: QConfigMapping | dict[str, Any] | None = ...,
    backend_config: BackendConfig | dict[str, Any] | None = ...,
    is_standalone_module: bool = ...,
) -> GraphModule: ...
