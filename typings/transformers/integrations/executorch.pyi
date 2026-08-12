from typing import Any
from collections.abc import Callable

import torch

from ..modeling_utils import PreTrainedModel

class TorchExportableModuleForVLM:
    def __init__(
        self, model, max_batch_size: int = ..., max_cache_len: int = ...
    ) -> None: ...
    def export_vision_encoder(self):  # -> ExportedProgram:
        ...
    def export_connector(self):  # -> ExportedProgram:
        ...
    def export_text_decoder(self):  # -> ExportedProgram:
        ...
    def export(self, **kwargs):  # -> dict[str, ExportedProgram | None]:
        ...
    def forward(self, pixel_values, input_ids, cache_position):  # -> None:
        ...
    def generate(
        self,
        pixel_values=...,
        input_ids=...,
        max_new_tokens=...,
        do_sample=...,
        temperature=...,
        **kwargs,
    ):  # -> None:
        ...

class TorchExportableModuleForDecoderOnlyLM(torch.nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int | None = ...,
        max_cache_len: int | None = ...,
        device: torch.device | None = ...,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def export(
        self,
        input_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
        dynamic_shapes: dict | None = ...,
        strict: bool | None = ...,
    ) -> torch.export.ExportedProgram: ...
    @staticmethod
    def generate(
        exported_program: torch.export.ExportedProgram,
        tokenizer,
        prompt: str,
        max_new_tokens: int = ...,
        do_sample: bool = ...,
        temperature: float = ...,
        top_k: int = ...,
        top_p: float = ...,
        device: str = ...,
    ) -> str: ...

class TorchExportableModuleWithStaticCache(torch.nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int | None = ...,
        max_cache_len: int | None = ...,
        device: torch.device | None = ...,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    @staticmethod
    def generate(
        exported_program: torch.export.ExportedProgram,
        prompt_token_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor: ...

class TorchExportableModuleWithHybridCache(torch.nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int | None = ...,
        max_cache_len: int | None = ...,
        device: torch.device | None = ...,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def convert_and_export_with_cache(
    model: PreTrainedModel,
    example_input_ids: torch.Tensor | None = ...,
    example_cache_position: torch.Tensor | None = ...,
    dynamic_shapes: dict | None = ...,
    strict: bool | None = ...,
):  # -> ExportedProgram:
    ...

class Seq2SeqLMEncoderExportableModule(torch.nn.Module):
    def __init__(self, encoder_model) -> None: ...
    def forward(self, input_ids): ...

class Seq2SeqLMDecoderExportableModuleWithStaticCache(torch.nn.Module):
    def __init__(self, model, max_static_cache_length, batch_size) -> None: ...
    def forward(self, decoder_input_ids, encoder_hidden_states, cache_position): ...

class Seq2SeqLMExportableModule(torch.nn.Module):
    def __init__(
        self,
        model,
        batch_size=...,
        max_hidden_seq_length=...,
        cache_implementation=...,
        max_cache_length=...,
    ) -> None: ...
    def export(
        self,
        encoder_input_ids=...,
        decoder_input_ids=...,
        encoder_hidden_states=...,
        cache_position=...,
    ):  # -> Self:
        ...
    def generate(self, prompt_token_ids, max_new_tokens):  # -> list[int]:
        ...

def export_with_dynamic_cache(
    model: PreTrainedModel,
    example_input_ids: torch.Tensor | None = ...,
    example_attention_mask: torch.Tensor | None = ...,
):  # -> ExportedProgram:
    ...
def register_dynamic_cache_export_support():  # -> None:
    ...
def sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = ...,
    mask_function: Callable | None = ...,
    attention_mask: torch.Tensor | None = ...,
    local_size: int | None = ...,
    allow_is_causal_skip: bool = ...,
    allow_torch_fix: bool = ...,
    **kwargs,
) -> torch.Tensor | None: ...
