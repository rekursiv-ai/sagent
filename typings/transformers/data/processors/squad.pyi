from .utils import DataProcessor
from ...tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

MULTI_SEP_TOKENS_TOKENIZERS_SET = ...
logger = ...

def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):  # -> list[Any]:
    ...
def squad_convert_example_to_features_init(
    tokenizer_for_convert: PreTrainedTokenizerBase,
):  # -> None:
    ...
def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy=...,
    return_dataset=...,
    threads=...,
    tqdm_enabled=...,
):  # -> tuple[list[Any], TensorDataset] | list[Any]:
    ...

class SquadProcessor(DataProcessor):
    train_file = ...
    dev_file = ...
    def get_examples_from_dataset(self, dataset, evaluate=...):  # -> list[Any]:
        ...
    def get_train_examples(self, data_dir, filename=...):  # -> list[Any]:
        ...
    def get_dev_examples(self, data_dir, filename=...):  # -> list[Any]:
        ...

class SquadV1Processor(SquadProcessor):
    train_file = ...
    dev_file = ...

class SquadV2Processor(SquadProcessor):
    train_file = ...
    dev_file = ...

class SquadExample:
    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=...,
        is_impossible=...,
    ) -> None: ...

class SquadFeatures:
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id: str | None = ...,
        encoding: BatchEncoding | None = ...,
    ) -> None: ...

class SquadResult:
    def __init__(
        self,
        unique_id,
        start_logits,
        end_logits,
        start_top_index=...,
        end_top_index=...,
        cls_logits=...,
    ) -> None: ...
