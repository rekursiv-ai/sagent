from .configuration_mt5 import MT5Config
from ..t5.modeling_tf_t5 import (
    TFT5EncoderModel,
    TFT5ForConditionalGeneration,
    TFT5Model,
)

"""Tensorflow mT5 model."""
logger = ...
_CONFIG_FOR_DOC = ...

class TFMT5Model(TFT5Model):
    model_type = ...
    config_class = MT5Config

class TFMT5ForConditionalGeneration(TFT5ForConditionalGeneration):
    model_type = ...
    config_class = MT5Config

class TFMT5EncoderModel(TFT5EncoderModel):
    model_type = ...
    config_class = MT5Config

__all__ = ["TFMT5EncoderModel", "TFMT5ForConditionalGeneration", "TFMT5Model"]
