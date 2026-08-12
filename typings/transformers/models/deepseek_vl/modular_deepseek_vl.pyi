from typing import Any
from torch import nn

import torch

from ..auto import AutoConfig
from ..idefics.modeling_idefics import (
    IdeficsBaseModelOutputWithPast,
    IdeficsCausalLMOutputWithPast,
)
from ..janus.image_processing_janus import JanusImageProcessor
from ..janus.image_processing_janus_fast import JanusImageProcessorFast
from ..janus.modeling_janus import (
    JanusForConditionalGeneration,
    JanusModel,
    JanusPreTrainedModel,
)
from ...configuration_utils import PretrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring

logger = ...

class DeepseekVLConfig(PretrainedConfig):
    def __init__(
        self,
        text_config: AutoConfig | None = ...,
        vision_config: AutoConfig | None = ...,
        image_token_id: int = ...,
        **kwargs,
    ) -> None: ...

class DeepseekVLBaseModelOutputWithPast(IdeficsBaseModelOutputWithPast): ...
class DeepseekVLCausalLMOutputWithPast(IdeficsCausalLMOutputWithPast): ...

class DeepseekVLAligner(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, vision_encodings: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class DeepseekVLPreTrainedModel(JanusPreTrainedModel):
    _no_split_modules = ...

@auto_docstring
class DeepseekVLModel(JanusModel):
    def __init__(self, config) -> None: ...

class DeepseekVLForConditionalGeneration(JanusForConditionalGeneration):
    def prepare_embeddings_for_image_generation(self): ...
    def decode_image_tokens(self): ...
    def generate(self): ...

class DeepseekVLImageProcessor(JanusImageProcessor):
    def __init__(self, **super_kwargs) -> None: ...
    def postprocess(self): ...
    def unnormalize(self): ...

class DeepseekVLImageProcessorFast(JanusImageProcessorFast):
    def __init__(self, **super_kwargs) -> None: ...
    def postprocess(self): ...

class DeepseekVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class DeepseekVLProcessor(ProcessorMixin):
    attributes = ...
    valid_kwargs = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self, image_processor, tokenizer, chat_template=..., num_image_tokens=...
    ) -> None: ...
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        images: ImageInput | None = ...,
        **kwargs: Unpack[DeepseekVLProcessorKwargs],
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = [
    "DeepseekVLConfig",
    "DeepseekVLForConditionalGeneration",
    "DeepseekVLImageProcessor",
    "DeepseekVLImageProcessorFast",
    "DeepseekVLModel",
    "DeepseekVLPreTrainedModel",
    "DeepseekVLProcessor",
]
