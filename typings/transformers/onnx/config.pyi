from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import dataclasses

from ..configuration_utils import PretrainedConfig
from ..feature_extraction_utils import FeatureExtractionMixin
from ..image_processing_utils import ImageProcessingMixin
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import TensorType

logger = ...
DEFAULT_ONNX_OPSET = ...
EXTERNAL_DATA_FORMAT_SIZE_LIMIT = ...

@dataclasses.dataclass
class PatchingSpec:
    o: Any
    name: str
    custom_op: Callable
    orig_op: Callable | None = ...
    op_wrapper: Callable | None = ...

class OnnxConfig(ABC):
    default_fixed_batch = ...
    default_fixed_sequence = ...
    default_fixed_num_choices = ...
    torch_onnx_minimum_version = ...
    _tasks_to_common_outputs = ...
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = ...,
        patching_specs: list[PatchingSpec] | None = ...,
    ) -> None: ...
    @classmethod
    def from_model_config(
        cls, config: PretrainedConfig, task: str = ...
    ) -> OnnxConfig: ...
    @property
    @abstractmethod
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def values_override(self) -> Mapping[str, Any] | None: ...
    @property
    def default_batch_size(self) -> int: ...
    @property
    def default_sequence_length(self) -> int: ...
    @property
    def default_num_choices(self) -> int: ...
    @property
    def default_onnx_opset(self) -> int: ...
    @property
    def atol_for_validation(self) -> float: ...
    @property
    def is_torch_support_available(self) -> bool: ...
    @staticmethod
    def use_external_data_format(num_parameters: int) -> bool: ...
    def generate_dummy_inputs(
        self,
        preprocessor: PreTrainedTokenizerBase
        | FeatureExtractionMixin
        | ImageProcessingMixin,
        batch_size: int = ...,
        seq_length: int = ...,
        num_choices: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
        num_channels: int = ...,
        image_width: int = ...,
        image_height: int = ...,
        sampling_rate: int = ...,
        time_duration: float = ...,
        frequency: int = ...,
        tokenizer: PreTrainedTokenizerBase | None = ...,
    ) -> Mapping[str, Any]: ...
    def generate_dummy_inputs_onnxruntime(
        self, reference_model_inputs: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...
    def patch_ops(self):  # -> None:
        ...
    def restore_ops(self):  # -> None:
        ...
    @classmethod
    def flatten_output_collection_property(
        cls, name: str, field: Iterable[Any]
    ) -> dict[str, Any]: ...

class OnnxConfigWithPast(OnnxConfig, ABC):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = ...,
        patching_specs: list[PatchingSpec] | None = ...,
        use_past: bool = ...,
    ) -> None: ...
    @classmethod
    def with_past(
        cls, config: PretrainedConfig, task: str = ...
    ) -> OnnxConfigWithPast: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def values_override(self) -> Mapping[str, Any] | None: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def num_attention_heads(self) -> int: ...
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
    ) -> Mapping[str, Any]: ...
    def fill_with_past_key_values_(
        self,
        inputs_or_outputs: Mapping[str, Mapping[int, str]],
        direction: str,
        inverted_values_shape: bool = ...,
    ):  # -> None:
        ...
    def flatten_output_collection_property(
        self, name: str, field: Iterable[Any]
    ) -> dict[str, Any]: ...

class OnnxSeq2SeqConfigWithPast(OnnxConfigWithPast):
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def num_layers(self) -> tuple[int, ...]: ...
    @property
    def num_attention_heads(self) -> tuple[int, ...]: ...
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizerBase | None,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
    ) -> Mapping[str, Any]: ...
    def fill_with_past_key_values_(
        self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str
    ):  # -> None:
        ...
