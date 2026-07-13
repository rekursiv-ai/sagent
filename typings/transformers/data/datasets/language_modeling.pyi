from torch.utils.data import Dataset

import torch

from ...tokenization_utils import PreTrainedTokenizer

logger = ...
DEPRECATION_WARNING = ...

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=...,
        cache_dir: str | None = ...,
    ) -> None: ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, i) -> torch.Tensor: ...

class LineByLineTextDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int
    ) -> None: ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, i) -> dict[str, torch.tensor]: ...

class LineByLineWithRefDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        ref_path: str,
    ) -> None: ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, i) -> dict[str, torch.tensor]: ...

class LineByLineWithSOPTextDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int
    ) -> None: ...
    def create_examples_from_document(
        self, document, block_size, tokenizer, short_seq_prob=...
    ):  # -> list[Any]:
        ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, i) -> dict[str, torch.tensor]: ...

class TextDatasetForNextSentencePrediction(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=...,
        short_seq_probability=...,
        nsp_probability=...,
    ) -> None: ...
    def create_examples_from_document(
        self, document: list[list[int]], doc_index: int, block_size: int
    ):  # -> None:
        ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, i):  # -> Any:
        ...
