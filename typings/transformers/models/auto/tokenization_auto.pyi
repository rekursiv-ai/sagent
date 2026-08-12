from typing import Any

import os

from .configuration_auto import replace_list_option_in_docstrings
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_tokenizers_available

"""Auto Tokenizer class."""
if is_tokenizers_available(): ...
logger = ...
TOKENIZER_MAPPING_NAMES = ...
TOKENIZER_MAPPING = ...
CONFIG_TO_TYPE = ...

def tokenizer_class_from_name(class_name: str) -> type[Any] | None: ...
def get_tokenizer_config(
    pretrained_model_name_or_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = ...,
    force_download: bool = ...,
    resume_download: bool | None = ...,
    proxies: dict[str, str] | None = ...,
    token: bool | str | None = ...,
    revision: str | None = ...,
    local_files_only: bool = ...,
    subfolder: str = ...,
    **kwargs,
) -> dict[str, Any]: ...

class AutoTokenizer:
    def __init__(self) -> None: ...
    @classmethod
    @replace_list_option_in_docstrings(TOKENIZER_MAPPING_NAMES)
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *inputs: object,
        **kwargs: object,
    ) -> PreTrainedTokenizerFast: ...
    @staticmethod
    def register(
        config_class, slow_tokenizer_class=..., fast_tokenizer_class=..., exist_ok=...
    ):  # -> None:
        ...

__all__ = ["TOKENIZER_MAPPING", "AutoTokenizer"]
