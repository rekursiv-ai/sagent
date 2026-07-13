from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings

"""BARK model configuration"""
logger = ...
BARK_SUBMODELCONFIG_START_DOCSTRING = ...

class BarkSubModelConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        block_size=...,
        input_vocab_size=...,
        output_vocab_size=...,
        num_layers=...,
        num_heads=...,
        hidden_size=...,
        dropout=...,
        bias=...,
        initializer_range=...,
        use_cache=...,
        **kwargs,
    ) -> None: ...

@add_start_docstrings(
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(
        config="BarkSemanticConfig", model="BarkSemanticModel"
    ),
    ...,
)
class BarkSemanticConfig(BarkSubModelConfig):
    model_type = ...
    base_config_key = ...

@add_start_docstrings(
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(
        config="BarkCoarseConfig", model="BarkCoarseModel"
    ),
    ...,
)
class BarkCoarseConfig(BarkSubModelConfig):
    model_type = ...
    base_config_key = ...

@add_start_docstrings(
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(
        config="BarkFineConfig", model="BarkFineModel"
    ),
    ...,
)
class BarkFineConfig(BarkSubModelConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self, tie_word_embeddings=..., n_codes_total=..., n_codes_given=..., **kwargs
    ) -> None: ...

class BarkConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        semantic_config: dict | None = ...,
        coarse_acoustics_config: dict | None = ...,
        fine_acoustics_config: dict | None = ...,
        codec_config: dict | None = ...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_sub_model_configs(
        cls,
        semantic_config: BarkSemanticConfig,
        coarse_acoustics_config: BarkCoarseConfig,
        fine_acoustics_config: BarkFineConfig,
        codec_config: PretrainedConfig,
        **kwargs,
    ):  # -> Self:
        ...

__all__ = ["BarkCoarseConfig", "BarkConfig", "BarkFineConfig", "BarkSemanticConfig"]
