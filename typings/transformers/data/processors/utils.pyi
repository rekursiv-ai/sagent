from dataclasses import dataclass

logger = ...

@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: str | None = ...
    label: str | None = ...
    def to_json_string(self):  # -> str:
        ...

@dataclass(frozen=True)
class InputFeatures:
    input_ids: list[int]
    attention_mask: list[int] | None = ...
    token_type_ids: list[int] | None = ...
    label: int | float | None = ...
    def to_json_string(self):  # -> str:
        ...

class DataProcessor:
    def get_example_from_tensor_dict(self, tensor_dict): ...
    def get_train_examples(self, data_dir): ...
    def get_dev_examples(self, data_dir): ...
    def get_test_examples(self, data_dir): ...
    def get_labels(self): ...
    def tfds_map(self, example): ...

class SingleSentenceClassificationProcessor(DataProcessor):
    def __init__(self, labels=..., examples=..., mode=..., verbose=...) -> None: ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, idx):  # -> SingleSentenceClassificationProcessor:
        ...
    @classmethod
    def create_from_csv(
        cls,
        file_name,
        split_name=...,
        column_label=...,
        column_text=...,
        column_id=...,
        skip_first_row=...,
        **kwargs,
    ):  # -> Self:
        ...
    @classmethod
    def create_from_examples(
        cls, texts_or_text_and_labels, labels=..., **kwargs
    ):  # -> Self:
        ...
    def add_examples_from_csv(
        self,
        file_name,
        split_name=...,
        column_label=...,
        column_text=...,
        column_id=...,
        skip_first_row=...,
        overwrite_labels=...,
        overwrite_examples=...,
    ):  # -> list[Any]:
        ...
    def add_examples(
        self,
        texts_or_text_and_labels,
        labels=...,
        ids=...,
        overwrite_labels=...,
        overwrite_examples=...,
    ):  # -> list[Any]:
        ...
    def get_features(
        self,
        tokenizer,
        max_length=...,
        pad_on_left=...,
        pad_token=...,
        mask_padding_with_zero=...,
        return_tensors=...,
    ):  # -> list[Any] | TensorDataset:
        ...
