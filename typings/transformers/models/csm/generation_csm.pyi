from dataclasses import dataclass

import torch

from ...generation import GenerateDecoderOnlyOutput, GenerationConfig, GenerationMixin
from ...generation.logits_process import LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateNonBeamOutput

logger = ...

@dataclass
class CsmGenerateOutput(GenerateDecoderOnlyOutput):
    audio: list[torch.Tensor] | None = ...

class CsmGenerationMixin(GenerationMixin):
    def generate(
        self,
        input_ids: torch.Tensor | None = ...,
        input_values: torch.Tensor | None = ...,
        input_values_cutoffs: torch.Tensor | None = ...,
        generation_config: GenerationConfig | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
        stopping_criteria: StoppingCriteriaList | None = ...,
        synced_gpus: bool | None = ...,
        streamer: BaseStreamer | None = ...,
        output_audio: bool | None = ...,
        **kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor: ...
