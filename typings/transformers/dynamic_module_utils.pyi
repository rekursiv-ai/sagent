from typing import Any

import os

"""Utilities to dynamically load objects from the Hub."""
logger = ...
_HF_REMOTE_CODE_LOCK = ...

def init_hf_modules():  # -> None:
    ...
def create_dynamic_module(name: str | os.PathLike) -> None: ...
def get_relative_imports(module_file: str | os.PathLike) -> list[str]: ...
def get_relative_import_files(module_file: str | os.PathLike) -> list[str]: ...
def get_imports(filename: str | os.PathLike) -> list[str]: ...
def check_imports(filename: str | os.PathLike) -> list[str]: ...
def get_class_in_module(
    class_name: str, module_path: str | os.PathLike, *, force_reload: bool = ...
) -> type: ...
def get_cached_module_file(
    pretrained_model_name_or_path: str | os.PathLike,
    module_file: str,
    cache_dir: str | os.PathLike | None = ...,
    force_download: bool = ...,
    resume_download: bool | None = ...,
    proxies: dict[str, str] | None = ...,
    token: bool | str | None = ...,
    revision: str | None = ...,
    local_files_only: bool = ...,
    repo_type: str | None = ...,
    _commit_hash: str | None = ...,
    **deprecated_kwargs,
) -> str: ...
def get_class_from_dynamic_module(
    class_reference: str,
    pretrained_model_name_or_path: str | os.PathLike,
    cache_dir: str | os.PathLike | None = ...,
    force_download: bool = ...,
    resume_download: bool | None = ...,
    proxies: dict[str, str] | None = ...,
    token: bool | str | None = ...,
    revision: str | None = ...,
    local_files_only: bool = ...,
    repo_type: str | None = ...,
    code_revision: str | None = ...,
    **kwargs,
) -> type: ...
def custom_object_save(
    obj: Any, folder: str | os.PathLike, config: dict | None = ...
) -> list[str]: ...

TIME_OUT_REMOTE_CODE = ...

def resolve_trust_remote_code(
    trust_remote_code,
    model_name,
    has_local_code,
    has_remote_code,
    error_message=...,
    upstream_repo=...,
):  # -> bool:
    ...
def check_python_requirements(
    path_or_repo_id, requirements_file=..., **kwargs
):  # -> None:
    ...
