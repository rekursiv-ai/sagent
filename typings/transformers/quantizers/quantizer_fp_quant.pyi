import torch

from .base import HfQuantizer
from ..modeling_utils import PreTrainedModel
from ..utils.quantization_config import QuantizationConfigMixin

logger = ...

class FPQuantHfQuantizer(HfQuantizer):
    requires_calibration = ...
    requires_parameters_quantization = ...
    required_packages = ...
    def __init__(
        self, quantization_config: QuantizationConfigMixin, **kwargs
    ) -> None: ...
    def validate_environment(self, device_map, **kwargs):  # -> None:
        ...
    def update_dtype(self, dtype: torch.dtype) -> torch.dtype: ...
    def create_quantized_param(
        self,
        model: PreTrainedModel,
        param_value: torch.Tensor,
        param_name: str,
        target_device: torch.device,
        **kwargs,
    ):  # -> None:
        ...
    def update_missing_keys(
        self, model, missing_keys: list[str], prefix: str
    ) -> list[str]: ...
    @property
    def is_trainable(self, model: PreTrainedModel | None = ...): ...  # noqa: PLR0206 -- mirrors upstream property definition
    def is_serializable(self, safe_serialization=...):  # -> Literal[True]:
        ...
    def param_needs_quantization(
        self, model: PreTrainedModel, param_name: str, **kwargs
    ) -> bool: ...
