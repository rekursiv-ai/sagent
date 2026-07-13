from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import os

import tensorflow as tf
import torch

from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..models.auto import AutoConfig
from ..processing_utils import ProcessorMixin
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
    ModelOutput,
    PushToHubMixin,
    add_end_docstrings,
    is_torch_available,
)
from ..utils.deprecation import deprecate_kwarg

type GenericTensor = list[GenericTensor] | torch.Tensor | tf.Tensor
if is_torch_available() or TYPE_CHECKING: ...
else: ...
logger = ...

def no_collate_fn(items): ...
def pad_collate_fn(tokenizer, feature_extractor):  # -> Callable[..., dict[Any, Any]]:
    ...
def infer_framework_load_model(
    model,
    config: AutoConfig,
    model_classes: dict[str, tuple[type]] | None = ...,
    task: str | None = ...,
    framework: str | None = ...,
    **model_kwargs,
): ...
def infer_framework_from_model(
    model,
    model_classes: dict[str, tuple[type]] | None = ...,
    task: str | None = ...,
    framework: str | None = ...,
    **model_kwargs,
): ...
def get_framework(model, revision: str | None = ...):  # -> Literal['tf', 'pt', 'flax']:
    ...
def get_default_model_and_revision(
    targeted_task: dict, framework: str | None, task_options: Any | None
) -> tuple[str, str]: ...
def load_assistant_model(
    model: PreTrainedModel,
    assistant_model: str | PreTrainedModel | None,
    assistant_tokenizer: PreTrainedTokenizer | None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizer | None]: ...

class PipelineException(Exception):
    def __init__(self, task: str, model: str, reason: str) -> None: ...

class ArgumentHandler(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs): ...

class PipelineDataFormat:
    SUPPORTED_FORMATS = ...
    def __init__(
        self,
        output_path: str | None,
        input_path: str | None,
        column: str | None,
        overwrite: bool = ...,
    ) -> None: ...
    @abstractmethod
    def __iter__(self): ...
    @abstractmethod
    def save(self, data: dict | list[dict]): ...
    def save_binary(self, data: dict | list[dict]) -> str: ...
    @staticmethod
    def from_str(
        format: str,
        output_path: str | None,
        input_path: str | None,
        column: str | None,
        overwrite=...,
    ) -> PipelineDataFormat: ...

class CsvPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self,
        output_path: str | None,
        input_path: str | None,
        column: str | None,
        overwrite=...,
    ) -> None: ...
    def __iter__(self):  # -> Generator[dict[str, str | Any] | str | Any, Any, None]:
        ...
    def save(self, data: list[dict]):  # -> None:
        ...

class JsonPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self,
        output_path: str | None,
        input_path: str | None,
        column: str | None,
        overwrite=...,
    ) -> None: ...
    def __iter__(self):  # -> Generator[dict[str, Any] | Any, Any, None]:
        ...
    def save(self, data: dict):  # -> None:
        ...

class PipedPipelineDataFormat(PipelineDataFormat):
    def __iter__(
        self,
    ):  # -> Generator[dict[str, str] | tuple[str, ...] | str | Any, Any, None]:
        ...
    def save(self, data: dict):  # -> None:
        ...
    def save_binary(self, data: dict | list[dict]) -> str: ...

class _ScikitCompat(ABC):
    @abstractmethod
    def transform(self, X): ...
    @abstractmethod
    def predict(self, X): ...

def build_pipeline_init_args(
    has_tokenizer: bool = ...,
    has_feature_extractor: bool = ...,
    has_image_processor: bool = ...,
    has_processor: bool = ...,
    supports_binary_output: bool = ...,
) -> str: ...

PIPELINE_INIT_ARGS = ...
SUPPORTED_PEFT_TASKS = ...

@add_end_docstrings(
    build_pipeline_init_args(
        has_tokenizer=True,
        has_feature_extractor=True,
        has_image_processor=True,
        has_processor=True,
    )
)
class Pipeline(_ScikitCompat, PushToHubMixin):
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    _pipeline_calls_generate = ...
    default_input_names = ...
    def __init__(
        self,
        model: PreTrainedModel | TFPreTrainedModel,
        tokenizer: PreTrainedTokenizer | None = ...,
        feature_extractor: PreTrainedFeatureExtractor | None = ...,
        image_processor: BaseImageProcessor | None = ...,
        processor: ProcessorMixin | None = ...,
        modelcard: ModelCard | None = ...,
        framework: str | None = ...,
        task: str = ...,
        device: int | torch.device | None = ...,
        binary_output: bool = ...,
        **kwargs,
    ) -> None: ...
    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        safe_serialization: bool = ...,
        **kwargs: Any,
    ):  # -> None:
        ...
    def transform(
        self, X
    ):  # -> list[Any] | PipelineIterator | Generator[Any, Any, None] | Tensor | Any | None:
        ...
    def predict(
        self, X
    ):  # -> list[Any] | PipelineIterator | Generator[Any, Any, None] | Tensor | Any | None:
        ...
    @property
    def dtype(self) -> torch.dtype | None: ...
    @property
    def torch_dtype(self) -> torch.dtype | None: ...
    @contextmanager
    def device_placement(self):  # -> Generator[None, Any, None]:
        ...
    def ensure_tensor_on_device(
        self, **inputs
    ):  # -> ModelOutput | dict[Any, Any] | UserDict[Any, Any] | list[Any] | tuple[Any, ...] | Tensor:
        ...
    def check_model_type(self, supported_models: list[str] | dict):  # -> None:
        ...
    @abstractmethod
    def preprocess(
        self, input_: Any, **preprocess_parameters: dict
    ) -> dict[str, GenericTensor]: ...
    @abstractmethod
    def postprocess(
        self, model_outputs: ModelOutput, **postprocess_parameters: dict
    ) -> Any: ...
    def get_inference_context(self):  # -> type[no_grad]:
        ...
    def forward(
        self, model_inputs, **forward_params
    ):  # -> ModelOutput | dict[Any, Any] | UserDict[Any, Any] | list[Any] | tuple[Any, ...] | Tensor:
        ...
    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params,
        forward_params,
        postprocess_params,
    ):  # -> PipelineIterator:
        ...
    def __call__(
        self, inputs, *args, num_workers=..., batch_size=..., **kwargs
    ):  # -> list[Any] | PipelineIterator | Generator[Any, Any, None] | Tensor | Any | None:
        ...
    def run_multi(
        self, inputs, preprocess_params, forward_params, postprocess_params
    ):  # -> list[Any]:
        ...
    def run_single(
        self, inputs, preprocess_params, forward_params, postprocess_params
    ):  # -> Any:
        ...
    def iterate(
        self, inputs, preprocess_params, forward_params, postprocess_params
    ):  # -> Generator[Any, Any, None]:
        ...

class ChunkPipeline(Pipeline):
    def run_single(
        self, inputs, preprocess_params, forward_params, postprocess_params
    ):  # -> Any:
        ...
    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params,
        forward_params,
        postprocess_params,
    ):  # -> PipelineIterator:
        ...

class PipelineRegistry:
    def __init__(
        self, supported_tasks: dict[str, Any], task_aliases: dict[str, str]
    ) -> None: ...
    def get_supported_tasks(self) -> list[str]: ...
    def check_task(self, task: str) -> tuple[str, dict, Any]: ...
    @deprecate_kwarg(old_name="tf_model", version="5.0.0")
    def register_pipeline(
        self,
        task: str,
        pipeline_class: type,
        pt_model: type | tuple[type] | None = ...,
        tf_model: type | tuple[type] | None = ...,
        default: dict | None = ...,
        type: str | None = ...,
    ) -> None: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...
