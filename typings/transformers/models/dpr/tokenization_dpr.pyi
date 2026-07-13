from ..bert.tokenization_bert import BertTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings

"""Tokenization classes for DPR."""
logger = ...
VOCAB_FILES_NAMES = ...

class DPRContextEncoderTokenizer(BertTokenizer):
    vocab_files_names = ...

class DPRQuestionEncoderTokenizer(BertTokenizer):
    vocab_files_names = ...

DPRSpanPrediction = ...
DPRReaderOutput = ...
CUSTOM_DPR_READER_DOCSTRING = ...

@add_start_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class CustomDPRReaderTokenizerMixin:
    def __call__(
        self,
        questions,
        titles: str | None = ...,
        texts: str | None = ...,
        padding: bool | str = ...,
        truncation: bool | str = ...,
        max_length: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_attention_mask: bool | None = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def decode_best_spans(
        self,
        reader_input: BatchEncoding,
        reader_output: DPRReaderOutput,
        num_spans: int = ...,
        max_answer_length: int = ...,
        num_spans_per_passage: int = ...,
    ) -> list[DPRSpanPrediction]: ...

@add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class DPRReaderTokenizer(CustomDPRReaderTokenizerMixin, BertTokenizer):
    vocab_files_names = ...
    model_input_names = ...

__all__ = [
    "DPRContextEncoderTokenizer",
    "DPRQuestionEncoderTokenizer",
    "DPRReaderOutput",
    "DPRReaderTokenizer",
]
