from collections import OrderedDict

import os

from .configuration_auto import replace_list_option_in_docstrings
from ...utils.import_utils import requires

"""AutoImageProcessor class."""
logger = ...
FORCE_FAST_IMAGE_PROCESSOR = ...
IMAGE_PROCESSOR_MAPPING_NAMES: OrderedDict[str, tuple[str | None, str | None]] = ...
IMAGE_PROCESSOR_MAPPING = ...

def get_image_processor_class_from_name(
    class_name: str,
):  # -> type[BaseImageProcessorFast] | Any | None:
    ...
def get_image_processor_config(
    pretrained_model_name_or_path: str | os.PathLike,
    cache_dir: str | os.PathLike | None = ...,
    force_download: bool = ...,
    resume_download: bool | None = ...,
    proxies: dict[str, str] | None = ...,
    token: bool | str | None = ...,
    revision: str | None = ...,
    local_files_only: bool = ...,
    **kwargs,
):  # -> dict[Any, Any] | Any:
    ...

@requires(backends=("vision",))
class AutoImageProcessor:
    def __init__(self) -> None: ...
    @classmethod
    @replace_list_option_in_docstrings(IMAGE_PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs): ...
    @staticmethod
    def register(
        config_class,
        image_processor_class=...,
        slow_image_processor_class=...,
        fast_image_processor_class=...,
        exist_ok=...,
    ):  # -> None:
        ...

__all__ = ["IMAGE_PROCESSOR_MAPPING", "AutoImageProcessor"]
