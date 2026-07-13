from typing import Any

from pyctcdecode import BeamSearchDecoderCTC

import numpy as np
import torch

from .base import ChunkPipeline
from ..feature_extraction_sequence_utils import SequenceFeatureExtractor
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils import PreTrainedTokenizer

logger = ...

def rescale_stride(stride, ratio):  # -> list[Any]:
    ...
def chunk_iter(
    inputs, feature_extractor, chunk_len, stride_left, stride_right, dtype=...
):  # -> Generator[dict[str | Any, Any | tuple[Any, Any | Literal[0], Any | Literal[0]]], Any, None]:
    ...

class AutomaticSpeechRecognitionPipeline(ChunkPipeline):
    _pipeline_calls_generate = ...
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    _default_generation_config = ...
    def __init__(
        self,
        model: PreTrainedModel,
        feature_extractor: SequenceFeatureExtractor | str | None = ...,
        tokenizer: PreTrainedTokenizer | None = ...,
        decoder: BeamSearchDecoderCTC | str | None = ...,
        device: int | torch.device | None = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self, inputs: np.ndarray | bytes | str | dict, **kwargs: Any
    ) -> list[dict[str, Any]]: ...
    def preprocess(self, inputs, chunk_length_s=..., stride_length_s=...): ...
    def postprocess(
        self,
        model_outputs,
        decoder_kwargs: dict | None = ...,
        return_timestamps=...,
        return_language=...,
    ):  # -> dict[str | Any, Any | str | list[Any]]:
        ...
