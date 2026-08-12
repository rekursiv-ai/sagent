from collections.abc import Callable, Iterator
from typing import Any

from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

import datasets
import optuna
import torch

from .data.data_collator import DataCollator
from .feature_extraction_utils import FeatureExtractionMixin
from .image_processing_utils import BaseImageProcessor
from .modeling_utils import PreTrainedModel
from .processing_utils import ProcessorMixin
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import TrainerCallback
from .trainer_utils import (
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
)
from .training_args import TrainingArguments
from .utils import (
    is_accelerate_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
)
from .utils.deprecation import deprecate_kwarg
from .utils.import_utils import requires

"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
DEFAULT_CALLBACKS = ...
DEFAULT_PROGRESS_CALLBACK = ...
if is_in_notebook():
    DEFAULT_PROGRESS_CALLBACK = ...
if is_torch_xla_available():
    IS_XLA_FSDPV2_POST_2_2 = ...
else:
    IS_XLA_FSDPV2_POST_2_2 = ...
if is_sagemaker_mp_enabled():
    IS_SAGEMAKER_MP_POST_1_10 = ...
else:
    IS_SAGEMAKER_MP_POST_1_10 = ...
if is_accelerate_available():
    DATA_SAMPLERS = ...

def safe_globals():  # -> nullcontext[None] | safe_globals:
    ...

logger = ...
TRAINING_ARGS_NAME = ...
TRAINER_STATE_NAME = ...
OPTIMIZER_NAME = ...
SCALER_NAME = ...
OPTIMIZER_NAME_BIN = ...
SCHEDULER_NAME = ...
FSDP_MODEL_NAME = ...

@requires(backends=("torch", "accelerate"))
class Trainer:
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
        eval_dataset: Dataset | dict[str, Dataset] | datasets.Dataset | None = ...,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = ...,
        model_init: Callable[..., PreTrainedModel] | None = ...,
        compute_loss_func: Callable | None = ...,
        compute_metrics: Callable[[EvalPrediction], dict] | None = ...,
        callbacks: list[TrainerCallback] | None = ...,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = ...,
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]]
        | None = ...,
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = ...,
    ) -> None: ...
    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None: ...
    @tokenizer.setter
    def tokenizer(self, processing_class) -> None: ...
    def add_callback(self, callback):  # -> None:
        ...
    def pop_callback(self, callback):  # -> None:
        ...
    def remove_callback(self, callback):  # -> None:
        ...
    def get_train_dataloader(self) -> DataLoader: ...
    def get_eval_dataloader(
        self, eval_dataset: str | Dataset | None = ...
    ) -> DataLoader: ...
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader: ...
    def create_optimizer_and_scheduler(self, num_training_steps: int):  # -> None:
        ...
    def get_decay_parameter_names(self, model) -> list[str]: ...
    def create_optimizer(self):  # -> Optimizer | Any:
        ...
    def get_num_trainable_parameters(self):  # -> int:
        ...
    def get_learning_rates(self):  # -> list[Any]:
        ...
    def get_optimizer_group(
        self, param: str | torch.nn.parameter.Parameter | None = ...
    ):  # -> dict[str, Any] | Any | list[Any]:
        ...
    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: PreTrainedModel | None = ...
    ) -> tuple[Any, Any]: ...
    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = ...
    ):  # -> LayerWiseDummyScheduler | ReduceLROnPlateau | LambdaLR:
        ...
    def num_examples(self, dataloader: DataLoader) -> int: ...
    @staticmethod
    def num_tokens(train_dl: DataLoader, max_steps: int | None = ...) -> int: ...
    def call_model_init(self, trial=...):  # -> PreTrainedModel:
        ...
    def torch_jit_model_eval(
        self, model, dataloader, training=...
    ):  # -> RecursiveScriptModule:
        ...
    def compare_trainer_and_checkpoint_args(
        self, training_args, trainer_state
    ):  # -> None:
        ...
    def train(
        self,
        resume_from_checkpoint: str | bool | None = ...,
        trial: optuna.Trial | dict[str, Any] | None = ...,
        ignore_keys_for_eval: list[str] | None = ...,
        **kwargs: Any,
    ): ...
    def get_tp_size(self) -> int: ...
    def get_total_train_batch_size(self, args) -> int: ...
    def hyperparameter_search(
        self,
        hp_space: Callable[[optuna.Trial], dict[str, float]] | None = ...,
        compute_objective: Callable[[dict[str, float]], float] | None = ...,
        n_trials: int = ...,
        direction: str | list[str] = ...,
        backend: str | HPSearchBackend | None = ...,
        hp_name: Callable[[optuna.Trial], str] | None = ...,
        **kwargs,
    ) -> BestRun | list[BestRun]: ...
    def log(self, logs: dict[str, float], start_time: float | None = ...) -> None: ...
    def compute_loss_context_manager(self):  # -> ExitStack[bool | None]:
        ...
    def autocast_smart_context_manager(
        self, cache_enabled: bool | None = ...
    ):  # -> autocast | nullcontext[None]:
        ...
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = ...,
        num_items_in_batch: torch.Tensor | None = ...,
    ):  # -> tuple[Any | Tensor, Any | dict[Any, Any]] | Tensor | Any:
        ...
    def is_local_process_zero(self) -> bool: ...
    def is_world_process_zero(self) -> bool: ...
    def save_model(
        self, output_dir: str | None = ..., _internal_call: bool = ...
    ):  # -> None:
        ...
    def store_flos(self):  # -> None:
        ...
    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = ...,
        ignore_keys: list[str] | None = ...,
        metric_key_prefix: str = ...,
    ) -> dict[str, float]: ...
    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: list[str] | None = ...,
        metric_key_prefix: str = ...,
    ) -> PredictionOutput: ...
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = ...,
        ignore_keys: list[str] | None = ...,
        metric_key_prefix: str = ...,
    ) -> EvalLoopOutput: ...
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = ...,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]: ...
    def floating_point_ops(self, inputs: dict[str, torch.Tensor | Any]):  # -> int:
        ...
    def init_hf_repo(self, token: str | None = ...):  # -> None:
        ...
    def create_model_card(
        self,
        language: str | None = ...,
        license: str | None = ...,
        tags: str | list[str] | None = ...,
        model_name: str | None = ...,
        finetuned_from: str | None = ...,
        tasks: str | list[str] | None = ...,
        dataset_tags: str | list[str] | None = ...,
        dataset: str | list[str] | None = ...,
        dataset_args: str | list[str] | None = ...,
    ):  # -> None:
        ...
    def push_to_hub(
        self,
        commit_message: str | None = ...,
        blocking: bool = ...,
        token: str | None = ...,
        revision: str | None = ...,
        **kwargs,
    ) -> str: ...
    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = ...,
        ignore_keys: list[str] | None = ...,
        metric_key_prefix: str = ...,
    ) -> EvalLoopOutput: ...
    def create_accelerator_and_postprocess(self):  # -> None:
        ...
    def propagate_args_to_deepspeed(self, auto_find_batch_size=...):  # -> None:
        ...
    def get_batch_samples(
        self, epoch_iterator: Iterator, num_batches: int, device: torch.device
    ) -> tuple[list, torch.Tensor | int | None]: ...
    def set_initial_training_values(
        self,
        args: TrainingArguments,
        dataloader: DataLoader,
        total_train_batch_size: int,
    ):  # -> tuple[int, int | Any, int, int | float, bool, int | None, int]:
        ...
