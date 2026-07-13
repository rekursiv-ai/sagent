from concurrent import futures
from pathlib import Path

import os

"""
Hub utilities: utilities related to download and cache models
"""
LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE = ...
CHAT_TEMPLATE_FILE = ...
CHAT_TEMPLATE_DIR = ...
logger = ...
_is_offline_mode = ...

def is_offline_mode():  # -> bool:
    ...

torch_cache_home = ...
default_cache_path = ...
PYTORCH_PRETRAINED_BERT_CACHE = ...
PYTORCH_TRANSFORMERS_CACHE = ...
TRANSFORMERS_CACHE = ...
HF_MODULES_CACHE = ...
TRANSFORMERS_DYNAMIC_MODULE_NAME = ...
SESSION_ID = ...
S3_BUCKET_PREFIX = ...
CLOUDFRONT_DISTRIB_PREFIX = ...
_staging_mode = ...
_default_endpoint = ...
HUGGINGFACE_CO_RESOLVE_ENDPOINT = ...
if os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None) is not None:
    HUGGINGFACE_CO_RESOLVE_ENDPOINT = ...
HUGGINGFACE_CO_RESOLVE_ENDPOINT = ...
HUGGINGFACE_CO_PREFIX = ...
HUGGINGFACE_CO_EXAMPLES_TELEMETRY = ...

def list_repo_templates(
    repo_id: str,
    *,
    local_files_only: bool,
    revision: str | None = ...,
    cache_dir: str | None = ...,
    token: bool | str | None = ...,
) -> list[str]: ...
def is_remote_url(url_or_filename):  # -> bool:
    ...
def define_sagemaker_information():  # -> dict[str, str | bool | Any | None]:
    ...
def http_user_agent(user_agent: dict | str | None = ...) -> str: ...
def extract_commit_hash(
    resolved_file: str | None, commit_hash: str | None
) -> str | None: ...
def cached_file(
    path_or_repo_id: str | os.PathLike, filename: str, **kwargs
) -> str | None: ...
def cached_files(
    path_or_repo_id: str | os.PathLike,
    filenames: list[str],
    cache_dir: str | os.PathLike | None = ...,
    force_download: bool = ...,
    resume_download: bool | None = ...,
    proxies: dict[str, str] | None = ...,
    token: bool | str | None = ...,
    revision: str | None = ...,
    local_files_only: bool = ...,
    subfolder: str = ...,
    repo_type: str | None = ...,
    user_agent: str | dict[str, str] | None = ...,
    _raise_exceptions_for_gated_repo: bool = ...,
    _raise_exceptions_for_missing_entries: bool = ...,
    _raise_exceptions_for_connection_errors: bool = ...,
    _commit_hash: str | None = ...,
    **deprecated_kwargs,
) -> str | None: ...
def download_url(url, proxies=...):  # -> str:
    ...
def has_file(
    path_or_repo: str | os.PathLike,
    filename: str,
    revision: str | None = ...,
    proxies: dict[str, str] | None = ...,
    token: bool | str | None = ...,
    *,
    local_files_only: bool = ...,
    cache_dir: str | Path | None = ...,
    repo_type: str | None = ...,
    **deprecated_kwargs,
):  # -> bool:
    ...

class PushToHubMixin:
    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: bool | None = ...,
        commit_message: str | None = ...,
        private: bool | None = ...,
        token: bool | str | None = ...,
        max_shard_size: int | str | None = ...,
        create_pr: bool = ...,
        safe_serialization: bool = ...,
        revision: str | None = ...,
        commit_description: str | None = ...,
        tags: list[str] | None = ...,
        **deprecated_kwargs,
    ) -> str: ...

def convert_file_size_to_int(size: int | str):  # -> int:
    ...
def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=...,
    force_download=...,
    proxies=...,
    resume_download=...,
    local_files_only=...,
    token=...,
    user_agent=...,
    revision=...,
    subfolder=...,
    _commit_hash=...,
    **deprecated_kwargs,
):  # -> tuple[list[str], Any] | tuple[str | None, Any]:
    ...
def create_and_tag_model_card(
    repo_id: str,
    tags: list[str] | None = ...,
    token: str | None = ...,
    ignore_metadata_errors: bool = ...,
):  # -> ModelCard:
    ...

class PushInProgress:
    def __init__(self, jobs: futures.Future | None = ...) -> None: ...
    def is_done(self):  # -> bool:
        ...
    def wait_until_done(self):  # -> None:
        ...
    def cancel(self) -> None: ...
