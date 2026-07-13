from collections.abc import Callable

from tf_keras.optimizers.legacy import Adam

from .modeling_tf_utils import keras

"""Functions and classes related to optimization (weight updates)."""
if hasattr(keras.optimizers.schedules, "learning_rate_schedule"):
    schedules = ...
else:
    schedules = ...

class WarmUp(schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = ...,
        name: str | None = ...,
    ) -> None: ...
    def __call__(self, step): ...
    def get_config(
        self,
    ):  # -> dict[str, float | Callable[..., Any] | int | str | None]:
        ...

def create_optimizer(
    init_lr: float,
    num_train_steps: int,
    num_warmup_steps: int,
    min_lr_ratio: float = ...,
    adam_beta1: float = ...,
    adam_beta2: float = ...,
    adam_epsilon: float = ...,
    adam_clipnorm: float | None = ...,
    adam_global_clipnorm: float | None = ...,
    weight_decay_rate: float = ...,
    power: float = ...,
    include_in_weight_decay: list[str] | None = ...,
):  # -> tuple[AdamWeightDecay | Any, WarmUp | Any]:
    ...

class AdamWeightDecay(Adam):
    def __init__(
        self,
        learning_rate: float | schedules.LearningRateSchedule = ...,
        beta_1: float = ...,
        beta_2: float = ...,
        epsilon: float = ...,
        amsgrad: bool = ...,
        weight_decay_rate: float = ...,
        include_in_weight_decay: list[str] | None = ...,
        exclude_from_weight_decay: list[str] | None = ...,
        name: str = ...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_config(cls, config): ...
    def apply_gradients(self, grads_and_vars, name=..., **kwargs): ...
    def get_config(self): ...

class GradientAccumulator:
    def __init__(self) -> None: ...
    @property
    def step(self): ...
    @property
    def gradients(self):  # -> list[Any]:
        ...
    def __call__(self, gradients):  # -> None:
        ...
    def reset(self):  # -> None:
        ...
