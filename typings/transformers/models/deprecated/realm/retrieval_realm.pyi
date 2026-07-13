import os

import numpy as np

"""REALM Retriever model implementation."""
_REALM_BLOCK_RECORDS_FILENAME = ...
logger = ...

def convert_tfrecord_to_np(
    block_records_path: str, num_block_records: int
) -> np.ndarray: ...

class ScaNNSearcher:
    def __init__(
        self,
        db,
        num_neighbors,
        dimensions_per_block=...,
        num_leaves=...,
        num_leaves_to_search=...,
        training_sample_size=...,
    ) -> None: ...
    def search_batched(self, question_projection): ...

class RealmRetriever:
    def __init__(self, block_records, tokenizer) -> None: ...
    def __call__(
        self,
        retrieved_block_ids,
        question_input_ids,
        answer_ids,
        max_length=...,
        return_tensors=...,
    ):  # -> tuple[list[Any], list[Any], list[Any], Any] | tuple[None, None, None, Any]:
        ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *init_inputs,
        **kwargs,
    ):  # -> Self:
        ...
    def save_pretrained(self, save_directory):  # -> None:
        ...
    def block_has_answer(
        self, concat_inputs, answer_ids
    ):  # -> tuple[list[Any], list[Any], list[Any]]:
        ...

__all__ = ["RealmRetriever"]
