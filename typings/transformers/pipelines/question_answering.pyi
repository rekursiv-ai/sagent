from torch.utils.data import Dataset

import numpy as np

from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args
from ..data import SquadExample
from ..modelcard import ModelCard
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import add_end_docstrings, is_tf_available

logger = ...
if is_tf_available():
    Dataset = ...

def decode_spans(
    start: np.ndarray,
    end: np.ndarray,
    topk: int,
    max_answer_len: int,
    undesired_tokens: np.ndarray,
) -> tuple: ...
def select_starts_ends(
    start,
    end,
    p_mask,
    attention_mask,
    min_null_score=...,
    top_k=...,
    handle_impossible_answer=...,
    max_answer_len=...,
):  # -> tuple[Any, Any, Any, int]:
    ...

class QuestionAnsweringArgumentHandler(ArgumentHandler):
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    def normalize(self, item):  # -> SquadExample | list[SquadExample]:
        ...
    def __call__(
        self, *args, **kwargs
    ):  # -> GeneratorType[Any, Any, Any] | Dataset[Any] | list[dict[Any, Any]] | list[dict[str, str]]:
        ...

@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class QuestionAnsweringPipeline(ChunkPipeline):
    default_input_names = ...
    handle_impossible_answer = ...
    def __init__(
        self,
        model: PreTrainedModel | TFPreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        modelcard: ModelCard | None = ...,
        framework: str | None = ...,
        task: str = ...,
        **kwargs,
    ) -> None: ...
    @staticmethod
    def create_sample(
        question: str | list[str], context: str | list[str]
    ) -> SquadExample | list[SquadExample]: ...
    def __call__(
        self, *args, **kwargs
    ):  # -> list[Any] | PipelineIterator | Generator[Any, Any, None] | Tensor | Any | None:
        ...
    def preprocess(
        self,
        example,
        padding=...,
        doc_stride=...,
        max_question_len=...,
        max_seq_len=...,
    ):  # -> Generator[dict[str | Any, SquadExample | Any | bool], Any, None]:
        ...
    def postprocess(
        self,
        model_outputs,
        top_k=...,
        handle_impossible_answer=...,
        max_answer_len=...,
        align_to_words=...,
    ):  # -> list[Any]:
        ...
    def get_answer(self, answers: list[dict], target: str) -> dict | None: ...
    def get_indices(
        self,
        enc: tokenizers.Encoding,
        s: int,
        e: int,
        sequence_index: int,
        align_to_words: bool,
    ) -> tuple[int, int]: ...
    def span_to_answer(
        self, text: str, start: int, end: int
    ) -> dict[str, str | int]: ...
