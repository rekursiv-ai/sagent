from collections.abc import Callable
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import IntervalStrategy, PreTrainedTokenizerBase
from .modeling_tf_utils import keras

logger = ...

class KerasMetricCallback(keras.callbacks.Callback):
    def __init__(
        self,
        metric_fn: Callable,
        eval_dataset: tf.data.Dataset | np.ndarray | tf.Tensor | tuple | dict,
        output_cols: list[str] | None = ...,
        label_cols: list[str] | None = ...,
        batch_size: int | None = ...,
        predict_with_generate: bool = ...,
        use_xla_generation: bool = ...,
        generate_kwargs: dict | None = ...,
    ) -> None: ...
    def on_epoch_end(self, epoch, logs=...):  # -> None:
        ...

class PushToHubCallback(keras.callbacks.Callback):
    def __init__(
        self,
        output_dir: str | Path,
        save_strategy: str | IntervalStrategy = ...,
        save_steps: int | None = ...,
        tokenizer: PreTrainedTokenizerBase | None = ...,
        hub_model_id: str | None = ...,
        hub_token: str | None = ...,
        checkpoint: bool = ...,
        **model_card_args,
    ) -> None: ...
    def on_train_begin(self, logs=...):  # -> None:
        ...
    def on_train_batch_end(self, batch, logs=...):  # -> None:
        ...
    def on_epoch_end(self, epoch, logs=...):  # -> None:
        ...
    def on_train_end(self, logs=...):  # -> None:
        ...
