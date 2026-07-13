from dataclasses import dataclass
from typing import Any

from torch import nn

import numpy as np
import torch

from .configuration_utils import PretrainedConfig, WatermarkingConfig
from .logits_process import SynthIDTextWatermarkLogitsProcessor
from ..modeling_utils import PreTrainedModel
from ..utils import ModelOutput

logger = ...

@dataclass
class WatermarkDetectorOutput:
    num_tokens_scored: np.ndarray | None = ...
    num_green_tokens: np.ndarray | None = ...
    green_fraction: np.ndarray | None = ...
    z_score: np.ndarray | None = ...
    p_value: np.ndarray | None = ...
    prediction: np.ndarray | None = ...
    confidence: np.ndarray | None = ...

class WatermarkDetector:
    def __init__(
        self,
        model_config: PretrainedConfig,
        device: str,
        watermarking_config: WatermarkingConfig | dict,
        ignore_repeated_ngrams: bool = ...,
        max_cache_size: int = ...,
    ) -> None: ...
    def __call__(
        self,
        input_ids: torch.LongTensor,
        z_threshold: float = ...,
        return_dict: bool = ...,
    ) -> WatermarkDetectorOutput | np.ndarray: ...

class BayesianDetectorConfig(PretrainedConfig):
    def __init__(
        self, watermarking_depth: int | None = ..., base_rate: float = ..., **kwargs
    ) -> None: ...
    def set_detector_information(self, model_name, watermarking_config):  # -> None:
        ...

@dataclass
class BayesianWatermarkDetectorModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    posterior_probabilities: torch.FloatTensor | None = ...

class BayesianDetectorWatermarkedLikelihood(nn.Module):
    def __init__(self, watermarking_depth: int) -> None: ...
    def forward(self, g_values: torch.Tensor) -> torch.Tensor: ...

class BayesianDetectorModel(PreTrainedModel):
    config: BayesianDetectorConfig
    base_model_prefix = ...
    def __init__(self, config) -> None: ...
    def forward(
        self,
        g_values: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = ...,
        loss_batch_weight=...,
        return_dict=...,
    ) -> BayesianWatermarkDetectorModelOutput: ...

class SynthIDTextWatermarkDetector:
    def __init__(
        self,
        detector_module: BayesianDetectorModel,
        logits_processor: SynthIDTextWatermarkLogitsProcessor,
        tokenizer: Any,
    ) -> None: ...
    def __call__(self, tokenized_outputs: torch.Tensor):  # -> Any:
        ...
