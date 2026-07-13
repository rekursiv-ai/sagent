from collections.abc import Callable
from typing import Any

from torch import nn
from torch.utils.data import Dataset, IterableDataset

import datasets
import torch

from .data.data_collator import DataCollator
from .feature_extraction_utils import FeatureExtractionMixin
from .generation.configuration_utils import GenerationConfig
from .image_processing_utils import BaseImageProcessor
from .modeling_utils import PreTrainedModel
from .processing_utils import ProcessorMixin
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer import Trainer
from .trainer_callback import TrainerCallback
from .trainer_utils import EvalPrediction, PredictionOutput
from .training_args import TrainingArguments
from .utils.deprecation import deprecate_kwarg

logger = ...

class Seq2SeqTrainer(Trainer):
    @deprecate_kwarg(
        "tokenizer",
        new_name="processing_class",
        version="5.0.0",
        raise_if_both_names=True,
    )
    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = ...,
        args: TrainingArguments | None = ...,
        data_collator: DataCollator | None = ...,
        train_dataset: Dataset | IterableDataset | datasets.Dataset | None = ...,
        eval_dataset: Dataset | dict[str, Dataset] | None = ...,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = ...,
        model_init: Callable[[], PreTrainedModel] | None = ...,
        compute_loss_func: Callable | None = ...,
        compute_metrics: Callable[[EvalPrediction], dict] | None = ...,
        callbacks: list[TrainerCallback] | None = ...,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = ...,
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = ...,
    ) -> None: ...
    @staticmethod
    def load_generation_config(
        gen_config_arg: str | GenerationConfig,
    ) -> GenerationConfig: ...
    def evaluate(
        self,
        eval_dataset: Dataset | None = ...,
        ignore_keys: list[str] | None = ...,
        metric_key_prefix: str = ...,
        **gen_kwargs,
    ) -> dict[str, float]: ...
    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: list[str] | None = ...,
        metric_key_prefix: str = ...,
        **gen_kwargs,
    ) -> PredictionOutput: ...
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = ...,
        **gen_kwargs,
    ) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None]: ...
