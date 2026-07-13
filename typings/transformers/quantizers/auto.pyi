from ..utils.quantization_config import QuantizationConfigMixin

AUTO_QUANTIZER_MAPPING = ...
AUTO_QUANTIZATION_CONFIG_MAPPING = ...
logger = ...

class AutoQuantizationConfig:
    @classmethod
    def from_dict(
        cls, quantization_config_dict: dict
    ):  # -> tuple[AwqConfig, dict[str, Any]] | AwqConfig | tuple[BitsAndBytesConfig, dict[str, Any]] | BitsAndBytesConfig | tuple[EetqConfig, dict[str, Any]] | EetqConfig | tuple[GPTQConfig, dict[str, Any]] | GPTQConfig | tuple[AqlmConfig, dict[str, Any]] | AqlmConfig | tuple[QuantoConfig, dict[str, Any]] | QuantoConfig | tuple[QuarkConfig, dict[str, Any]] | QuarkConfig | tuple[FPQuantConfig, dict[str, Any]] | FPQuantConfig | tuple[CompressedTensorsConfig, dict[str, Any]] | CompressedTensorsConfig | tuple[FbgemmFp8Config, dict[str, Any]] | FbgemmFp8Config | tuple[HiggsConfig, dict[str, Any]] | HiggsConfig | tuple[BitNetQuantConfig, dict[str, Any]] | BitNetQuantConfig | tuple[VptqConfig, dict[str, Any]] | VptqConfig | tuple[SpQRConfig, dict[str, Any]] | SpQRConfig | tuple[FineGrainedFP8Config, dict[str, Any]] | FineGrainedFP8Config | tuple[AutoRoundConfig, dict[str, Any]] | AutoRoundConfig | tuple[Mxfp4Config, dict[str, Any]] | Mxfp4Config | TorchAoConfig | HqqConfig:
        ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ):  # -> tuple[AwqConfig, dict[str, Any]] | AwqConfig | tuple[BitsAndBytesConfig, dict[str, Any]] | BitsAndBytesConfig | tuple[EetqConfig, dict[str, Any]] | EetqConfig | tuple[GPTQConfig, dict[str, Any]] | GPTQConfig | tuple[AqlmConfig, dict[str, Any]] | AqlmConfig | tuple[QuantoConfig, dict[str, Any]] | QuantoConfig | tuple[QuarkConfig, dict[str, Any]] | QuarkConfig | tuple[FPQuantConfig, dict[str, Any]] | FPQuantConfig | tuple[CompressedTensorsConfig, dict[str, Any]] | CompressedTensorsConfig | tuple[FbgemmFp8Config, dict[str, Any]] | FbgemmFp8Config | tuple[HiggsConfig, dict[str, Any]] | HiggsConfig | tuple[BitNetQuantConfig, dict[str, Any]] | BitNetQuantConfig | tuple[VptqConfig, dict[str, Any]] | VptqConfig | tuple[SpQRConfig, dict[str, Any]] | SpQRConfig | tuple[FineGrainedFP8Config, dict[str, Any]] | FineGrainedFP8Config | tuple[AutoRoundConfig, dict[str, Any]] | AutoRoundConfig | tuple[Mxfp4Config, dict[str, Any]] | Mxfp4Config | TorchAoConfig | HqqConfig:
        ...

class AutoHfQuantizer:
    @classmethod
    def from_config(
        cls, quantization_config: QuantizationConfigMixin | dict, **kwargs
    ):  # -> AqlmHfQuantizer | AutoRoundQuantizer | CompressedTensorsHfQuantizer | FPQuantHfQuantizer | GptqHfQuantizer | HiggsHfQuantizer | QuantoHfQuantizer | SpQRHfQuantizer | VptqHfQuantizer | AwqQuantizer | BitNetHfQuantizer | Bnb4BitHfQuantizer | Bnb8BitHfQuantizer | EetqHfQuantizer | FbgemmFp8HfQuantizer | FineGrainedFP8HfQuantizer | HqqHfQuantizer | Mxfp4HfQuantizer | QuarkHfQuantizer | TorchAoHfQuantizer:
        ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ):  # -> AqlmHfQuantizer | AutoRoundQuantizer | CompressedTensorsHfQuantizer | FPQuantHfQuantizer | GptqHfQuantizer | HiggsHfQuantizer | QuantoHfQuantizer | SpQRHfQuantizer | VptqHfQuantizer | AwqQuantizer | BitNetHfQuantizer | Bnb4BitHfQuantizer | Bnb8BitHfQuantizer | EetqHfQuantizer | FbgemmFp8HfQuantizer | FineGrainedFP8HfQuantizer | HqqHfQuantizer | Mxfp4HfQuantizer | QuarkHfQuantizer | TorchAoHfQuantizer:
        ...
    @classmethod
    def merge_quantization_configs(
        cls,
        quantization_config: dict | QuantizationConfigMixin,
        quantization_config_from_args: QuantizationConfigMixin | None,
    ):  # -> GPTQConfig | AwqConfig | AutoRoundConfig | FbgemmFp8Config | CompressedTensorsConfig | dict[Any, Any] | QuantizationConfigMixin | Mxfp4Config:
        ...
    @staticmethod
    def supports_quant_method(quantization_config_dict):  # -> bool:
        ...

def register_quantization_config(
    method: str,
):  # -> Callable[..., type[QuantizationConfigMixin]]:
    ...
def register_quantizer(name: str):  # -> Callable[..., type[HfQuantizer]]:
    ...
def get_hf_quantizer(
    config,
    quantization_config,
    dtype,
    from_tf,
    from_flax,
    device_map,
    weights_only,
    user_agent,
):  # -> tuple[AqlmHfQuantizer | AutoRoundQuantizer | CompressedTensorsHfQuantizer | FPQuantHfQuantizer | GptqHfQuantizer | HiggsHfQuantizer | QuantoHfQuantizer | SpQRHfQuantizer | VptqHfQuantizer | AwqQuantizer | BitNetHfQuantizer | Bnb4BitHfQuantizer | Bnb8BitHfQuantizer | EetqHfQuantizer | FbgemmFp8HfQuantizer | FineGrainedFP8HfQuantizer | HqqHfQuantizer | Mxfp4HfQuantizer | QuarkHfQuantizer | TorchAoHfQuantizer | None, Any, dtype | Any, dict[str, Any] | dict[str, device] | dict[str, str] | dict[str, int] | Any | None]:
    ...
