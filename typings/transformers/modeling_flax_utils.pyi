from typing import Any

import os

from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_utils import PretrainedConfig
from .generation import FlaxGenerationMixin
from .utils import PushToHubMixin

logger = ...

def quick_gelu(x): ...

ACT2FN = ...

def flax_shard_checkpoint(
    params, max_shard_size=...
):  # -> tuple[dict[str, Any], None] | tuple[dict[Any, Any], dict[str, dict[str, Any | int]]]:
    ...

class FlaxPreTrainedModel(PushToHubMixin, FlaxGenerationMixin):
    config_class = ...
    base_model_prefix = ...
    main_input_name = ...
    _auto_class = ...
    _missing_keys = ...
    def __init__(
        self,
        config: PretrainedConfig,
        module: nn.Module,
        input_shape: tuple = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> dict: ...
    def enable_gradient_checkpointing(self): ...
    @property
    def framework(self) -> str: ...
    @property
    def config(self) -> PretrainedConfig: ...
    @property
    def module(self) -> nn.Module: ...
    @property
    def params(self) -> dict | FrozenDict: ...
    @property
    def required_params(self) -> set: ...
    @property
    def params_shape_tree(self) -> dict: ...
    @params.setter
    def params(self, params: dict | FrozenDict):  # -> None:
        ...
    def to_bf16(self, params: dict | FrozenDict, mask: Any = ...):  # -> Any:
        ...
    def to_fp32(self, params: dict | FrozenDict, mask: Any = ...):  # -> Any:
        ...
    def to_fp16(self, params: dict | FrozenDict, mask: Any = ...):  # -> Any:
        ...
    @classmethod
    def load_flax_weights(cls, resolved_archive_file): ...
    @classmethod
    def load_flax_sharded_weights(cls, shard_files): ...
    @classmethod
    def can_generate(cls) -> bool: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        dtype: jnp.dtype = ...,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = ...,
        cache_dir: str | os.PathLike | None = ...,
        ignore_mismatched_sizes: bool = ...,
        force_download: bool = ...,
        local_files_only: bool = ...,
        token: str | bool | None = ...,
        revision: str = ...,
        **kwargs,
    ):  # -> Self | tuple[Self, Any]:
        ...
    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        params=...,
        push_to_hub=...,
        max_shard_size=...,
        token: str | bool | None = ...,
        safe_serialization: bool = ...,
        **kwargs,
    ):  # -> None:
        ...
    @classmethod
    def register_for_auto_class(cls, auto_class=...):  # -> None:
        ...

def overwrite_call_docstring(model_class, docstring):  # -> None:
    ...
def append_call_sample_docstring(
    model_class,
    checkpoint,
    output_type,
    config_class,
    mask=...,
    revision=...,
    real_checkpoint=...,
):  # -> None:
    ...
def append_replace_return_docstrings(
    model_class, output_type, config_class
):  # -> None:
    ...
