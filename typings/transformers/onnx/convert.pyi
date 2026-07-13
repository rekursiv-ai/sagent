from collections.abc import Iterable
from pathlib import Path

from packaging.version import Version

from .config import OnnxConfig
from ..feature_extraction_utils import FeatureExtractionMixin
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..processing_utils import ProcessorMixin
from ..tokenization_utils import PreTrainedTokenizer

logger = ...
ORT_QUANTIZE_MINIMUM_VERSION = ...

def check_onnxruntime_requirements(minimum_version: Version):  # -> None:
    ...
def export_pytorch(
    preprocessor: PreTrainedTokenizer | FeatureExtractionMixin | ProcessorMixin,
    model: PreTrainedModel,
    config: OnnxConfig,
    opset: int,
    output: Path,
    tokenizer: PreTrainedTokenizer | None = ...,
    device: str = ...,
) -> tuple[list[str], list[str]]: ...
def export_tensorflow(
    preprocessor: PreTrainedTokenizer | FeatureExtractionMixin,
    model: TFPreTrainedModel,
    config: OnnxConfig,
    opset: int,
    output: Path,
    tokenizer: PreTrainedTokenizer | None = ...,
) -> tuple[list[str], list[str]]: ...
def export(
    preprocessor: PreTrainedTokenizer | FeatureExtractionMixin | ProcessorMixin,
    model: PreTrainedModel | TFPreTrainedModel,
    config: OnnxConfig,
    opset: int,
    output: Path,
    tokenizer: PreTrainedTokenizer | None = ...,
    device: str = ...,
) -> tuple[list[str], list[str]]: ...
def validate_model_outputs(
    config: OnnxConfig,
    preprocessor: PreTrainedTokenizer | FeatureExtractionMixin | ProcessorMixin,
    reference_model: PreTrainedModel | TFPreTrainedModel,
    onnx_model: Path,
    onnx_named_outputs: list[str],
    atol: float,
    tokenizer: PreTrainedTokenizer | None = ...,
):  # -> None:
    ...
def ensure_model_and_config_inputs_match(
    model: PreTrainedModel | TFPreTrainedModel, model_inputs: Iterable[str]
) -> tuple[bool, list[str]]: ...
