from collections.abc import Callable

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny

__all__ = ["QConfigMultiMapping"]
_QCONFIG_STYLE_TO_METHOD: dict[str, str] = ...

class QConfigMultiMapping:
    def __init__(self) -> None: ...
    def set_global(
        self, global_qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping: ...
    def set_object_type(
        self, object_type: Callable | str, qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping: ...
    def set_module_name_regex(
        self, module_name_regex: str, qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping: ...
    def set_module_name(
        self, module_name: str, qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping: ...
    def set_module_name_object_type_order(
        self,
        module_name: str,
        object_type: Callable,
        index: int,
        qconfig_list: list[QConfigAny],
    ) -> QConfigMultiMapping: ...
    @classmethod
    def from_list_qconfig_mapping(
        cls, qconfig_mapping_list: list[QConfigMapping]
    ) -> QConfigMultiMapping: ...
