from collections.abc import Callable

import torch

from ...generation import GenerationConfig, GenerationMixin
from ...generation.logits_process import LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput

logger = ...

class WhisperGenerationMixin(GenerationMixin):
    def generate(
        self,
        input_features: torch.Tensor | None = ...,
        generation_config: GenerationConfig | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
        stopping_criteria: StoppingCriteriaList | None = ...,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = ...,
        synced_gpus: bool = ...,
        return_timestamps: bool | None = ...,
        task: str | None = ...,
        language: str | list[str] | None = ...,
        is_multilingual: bool | None = ...,
        prompt_ids: torch.Tensor | None = ...,
        prompt_condition_type: str | None = ...,
        condition_on_prev_tokens: bool | None = ...,
        temperature: float | tuple[float, ...] | None = ...,
        compression_ratio_threshold: float | None = ...,
        logprob_threshold: float | None = ...,
        no_speech_threshold: float | None = ...,
        num_segment_frames: int | None = ...,
        attention_mask: torch.Tensor | None = ...,
        time_precision: float = ...,
        time_precision_features: float = ...,
        return_token_timestamps: bool | None = ...,
        return_segments: bool = ...,
        return_dict_in_generate: bool | None = ...,
        force_unique_generate_call: bool | None = ...,
        monitor_progress: Callable[[torch.Tensor], None] | None = ...,
        **kwargs,
    ):  # -> dict[str, Any] | tuple[Tensor, Tensor] | Tensor | dict[str, Tensor] | dict[str, tuple[Tensor, Tensor] | Tensor]:
        ...
    def generate_with_fallback(
        self,
        segment_input,
        decoder_input_ids,
        cur_bsz,
        seek,
        batch_idx_map,
        temperatures,
        generation_config,
        logits_processor,
        stopping_criteria,
        prefix_allowed_tokens_fn,
        synced_gpus,
        return_token_timestamps,
        do_condition_on_prev_tokens,
        is_shortform,
        batch_size,
        attention_mask,
        kwargs,
    ):  # -> tuple[list[None] | Any | Tensor, list[None] | Any | Tensor | list[dict[Any, list[Any] | tuple[tuple[Any, ...], ...] | EncoderDecoderCache | tuple[Any, ...] | Any | None]], list[bool], Any, type[GenerateDecoderOnlyOutput] | type[GenerateEncoderDecoderOutput] | type[GenerateBeamDecoderOnlyOutput] | type[GenerateBeamEncoderDecoderOutput] | type[LongTensor] | Any]:
        ...
    def detect_language(
        self,
        input_features: torch.FloatTensor | None = ...,
        encoder_outputs: torch.FloatTensor | BaseModelOutput | None = ...,
        generation_config: GenerationConfig | None = ...,
        num_segment_frames: int = ...,
    ) -> torch.Tensor: ...
