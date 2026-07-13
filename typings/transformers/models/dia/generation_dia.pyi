from collections.abc import Callable

import torch

from ...generation.logits_process import LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateOutput, GenerationConfig, GenerationMixin
from ...modeling_utils import PreTrainedModel

logger = ...

class DiaGenerationMixin(GenerationMixin):
    _uses_cfg = ...
    def prepare_inputs_for_generation(
        self, input_ids, encoder_outputs=..., decoder_delay_mask=..., **kwargs
    ):  # -> dict[Any, Any]:
        ...
    @staticmethod
    def apply_delay_mask(
        input_ids: torch.Tensor, pad_id: int, delay_mask: torch.Tensor | None
    ) -> torch.Tensor: ...
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = ...,
        generation_config: GenerationConfig | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
        stopping_criteria: StoppingCriteriaList | None = ...,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = ...,
        synced_gpus: bool | None = ...,
        assistant_model: PreTrainedModel | None = ...,
        streamer: BaseStreamer | None = ...,
        negative_prompt_ids: torch.Tensor | None = ...,
        negative_prompt_attention_mask: torch.Tensor | None = ...,
        use_model_defaults: bool | None = ...,
        custom_generate: str | None = ...,
        **kwargs,
    ) -> GenerateOutput | torch.LongTensor: ...
