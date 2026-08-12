from ...configuration_utils import PretrainedConfig

logger = ...

class SpeechEncoderDecoderConfig(PretrainedConfig):
    def __init__(self, **kwargs) -> None: ...
    @classmethod
    def from_encoder_decoder_configs(
        cls,
        encoder_config: PretrainedConfig,
        decoder_config: PretrainedConfig,
        **kwargs,
    ) -> PretrainedConfig: ...

__all__ = ["SpeechEncoderDecoderConfig"]
