from dataclasses import dataclass

from .training_args import TrainingArguments

"""
Callbacks to use with the Trainer class and customize the training loop.
"""
logger = ...

@dataclass
class TrainerState:
    epoch: float | None = ...
    global_step: int = ...
    max_steps: int = ...
    logging_steps: int = ...
    eval_steps: int = ...
    save_steps: int = ...
    train_batch_size: int | None = ...
    num_train_epochs: int = ...
    num_input_tokens_seen: int = ...
    total_flos: float = ...
    log_history: list[dict[str, float]] = ...
    best_metric: float | None = ...
    best_global_step: int | None = ...
    best_model_checkpoint: str | None = ...
    is_local_process_zero: bool = ...
    is_world_process_zero: bool = ...
    is_hyper_param_search: bool = ...
    trial_name: str | None = ...
    trial_params: dict[str, str | float | int | bool] | None = ...
    stateful_callbacks: list[TrainerCallback] | None = ...
    def __post_init__(self):  # -> None:
        ...
    def save_to_json(self, json_path: str):  # -> None:
        ...
    @classmethod
    def load_from_json(cls, json_path: str):  # -> Self:
        ...
    def compute_steps(self, args, max_steps):  # -> None:
        ...
    def init_training_references(
        self, trainer, max_steps, num_train_epochs, trial
    ):  # -> None:
        ...

class ExportableState:
    def state(self) -> dict: ...
    @classmethod
    def from_state(cls, state):  # -> Self:
        ...

@dataclass
class TrainerControl(ExportableState):
    should_training_stop: bool = ...
    should_epoch_stop: bool = ...
    should_save: bool = ...
    should_evaluate: bool = ...
    should_log: bool = ...
    def state(self) -> dict: ...

class TrainerCallback:
    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):  # -> None:
        ...
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...
    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> None:
        ...

class CallbackHandler(TrainerCallback):
    def __init__(
        self, callbacks, model, processing_class, optimizer, lr_scheduler
    ) -> None: ...
    def add_callback(self, callback):  # -> None:
        ...
    def pop_callback(self, callback):  # -> None:
        ...
    def remove_callback(self, callback):  # -> None:
        ...
    @property
    def callback_list(self):  # -> LiteralString:
        ...
    def on_init_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_pre_optimizer_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_optimizer_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_substep_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
    ):  # -> Any:
        ...
    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
    ):  # -> Any:
        ...
    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs,
    ):  # -> Any:
        ...
    def on_prediction_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):  # -> Any:
        ...
    def call_event(self, event, args, state, control, **kwargs):  # -> Any:
        ...

class DefaultFlowCallback(TrainerCallback):
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> TrainerControl:
        ...
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # -> TrainerControl:
        ...

class ProgressCallback(TrainerCallback):
    def __init__(self, max_str_len: int = ...) -> None: ...
    def on_train_begin(self, args, state, control, **kwargs):  # -> None:
        ...
    def on_step_end(self, args, state, control, **kwargs):  # -> None:
        ...
    def on_prediction_step(
        self, args, state, control, eval_dataloader=..., **kwargs
    ):  # -> None:
        ...
    def on_evaluate(self, args, state, control, **kwargs):  # -> None:
        ...
    def on_predict(self, args, state, control, **kwargs):  # -> None:
        ...
    def on_log(self, args, state, control, logs=..., **kwargs):  # -> None:
        ...
    def on_train_end(self, args, state, control, **kwargs):  # -> None:
        ...

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=..., **kwargs):  # -> None:
        ...

class EarlyStoppingCallback(TrainerCallback, ExportableState):
    def __init__(
        self,
        early_stopping_patience: int = ...,
        early_stopping_threshold: float | None = ...,
    ) -> None: ...
    def check_metric_value(self, args, state, control, metric_value):  # -> None:
        ...
    def on_train_begin(self, args, state, control, **kwargs):  # -> None:
        ...
    def on_evaluate(self, args, state, control, metrics, **kwargs):  # -> None:
        ...
    def state(self) -> dict: ...
