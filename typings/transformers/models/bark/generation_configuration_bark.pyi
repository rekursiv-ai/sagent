from ...generation.configuration_utils import GenerationConfig

"""BARK model generation configuration"""
logger = ...

class BarkSemanticGenerationConfig(GenerationConfig):
    model_type = ...
    def __init__(
        self,
        eos_token_id=...,
        renormalize_logits=...,
        max_new_tokens=...,
        output_scores=...,
        return_dict_in_generate=...,
        output_hidden_states=...,
        output_attentions=...,
        temperature=...,
        do_sample=...,
        text_encoding_offset=...,
        text_pad_token=...,
        semantic_infer_token=...,
        semantic_vocab_size=...,
        max_input_semantic_length=...,
        semantic_rate_hz=...,
        min_eos_p=...,
        **kwargs,
    ) -> None: ...

class BarkCoarseGenerationConfig(GenerationConfig):
    model_type = ...
    def __init__(
        self,
        renormalize_logits=...,
        output_scores=...,
        return_dict_in_generate=...,
        output_hidden_states=...,
        output_attentions=...,
        temperature=...,
        do_sample=...,
        coarse_semantic_pad_token=...,
        coarse_rate_hz=...,
        n_coarse_codebooks=...,
        coarse_infer_token=...,
        max_coarse_input_length=...,
        max_coarse_history: int = ...,
        sliding_window_len: int = ...,
        **kwargs,
    ) -> None: ...

class BarkFineGenerationConfig(GenerationConfig):
    model_type = ...
    def __init__(
        self,
        temperature=...,
        max_fine_history_length=...,
        max_fine_input_length=...,
        n_fine_codebooks=...,
        **kwargs,
    ) -> None: ...
    def validate(self, **kwargs):  # -> None:
        ...

class BarkGenerationConfig(GenerationConfig):
    model_type = ...
    def __init__(
        self,
        semantic_config: dict | None = ...,
        coarse_acoustics_config: dict | None = ...,
        fine_acoustics_config: dict | None = ...,
        sample_rate=...,
        codebook_size=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_sub_model_configs(
        cls,
        semantic_config: BarkSemanticGenerationConfig,
        coarse_acoustics_config: BarkCoarseGenerationConfig,
        fine_acoustics_config: BarkFineGenerationConfig,
        **kwargs,
    ):  # -> Self:
        ...
    def to_dict(self):  # -> dict[str, Any]:
        ...
