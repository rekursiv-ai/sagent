from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any

import os

logger = ...

class QuantizationMethod(StrEnum):
    BITS_AND_BYTES = ...
    GPTQ = ...
    AWQ = ...
    AQLM = ...
    VPTQ = ...
    QUANTO = ...
    EETQ = ...
    HIGGS = ...
    HQQ = ...
    COMPRESSED_TENSORS = ...
    FBGEMM_FP8 = ...
    TORCHAO = ...
    BITNET = ...
    SPQR = ...
    FP8 = ...
    QUARK = ...
    FPQUANT = ...
    AUTOROUND = ...
    MXFP4 = ...

class AWQLinearVersion(StrEnum):
    GEMM = ...
    GEMV = ...
    EXLLAMA = ...
    IPEX = ...
    @staticmethod
    def from_str(
        version: str,
    ):  # -> Literal[AWQLinearVersion.GEMM, AWQLinearVersion.GEMV, AWQLinearVersion.EXLLAMA, AWQLinearVersion.IPEX]:
        ...

class AwqBackendPackingMethod(StrEnum):
    AUTOAWQ = ...
    LLMAWQ = ...

@dataclass
class QuantizationConfigMixin:
    quant_method: QuantizationMethod
    @classmethod
    def from_dict(
        cls, config_dict, return_unused_kwargs=..., **kwargs
    ):  # -> tuple[Self, dict[str, Any]] | Self:
        ...
    def to_json_file(self, json_file_path: str | os.PathLike):  # -> None:
        ...
    def to_dict(self) -> dict[str, Any]: ...
    def __iter__(self):  # -> Generator[tuple[str, Any], Any, None]:
        ...
    def __repr__(self):  # -> str:
        ...
    def to_json_string(self, use_diff: bool = ...) -> str: ...
    def update(self, **kwargs):  # -> dict[str, Any]:
        ...

@dataclass
class AutoRoundConfig(QuantizationConfigMixin):
    def __init__(
        self,
        bits: int = ...,
        group_size: int = ...,
        sym: bool = ...,
        backend: str = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...
    def get_loading_attributes(self):  # -> dict[str, str]:
        ...
    def to_dict(self):  # -> dict[str, Any]:
        ...
    @classmethod
    def from_dict(
        cls, config_dict, return_unused_kwargs=..., **kwargs
    ):  # -> tuple[Self, dict[str, Any]] | Self:
        ...

@dataclass
class HqqConfig(QuantizationConfigMixin):
    def __init__(
        self,
        nbits: int = ...,
        group_size: int = ...,
        view_as_float: bool = ...,
        axis: int | None = ...,
        dynamic_config: dict | None = ...,
        skip_modules: list[str] = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...
    @classmethod
    def from_dict(cls, config: dict[str, Any]):  # -> Self:
        ...
    def to_dict(self) -> dict[str, Any]: ...
    def __repr__(self):  # -> str:
        ...
    def to_diff_dict(self) -> dict[str, Any]: ...

@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    def __init__(
        self,
        load_in_8bit=...,
        load_in_4bit=...,
        llm_int8_threshold=...,
        llm_int8_skip_modules=...,
        llm_int8_enable_fp32_cpu_offload=...,
        llm_int8_has_fp16_weight=...,
        bnb_4bit_compute_dtype=...,
        bnb_4bit_quant_type=...,
        bnb_4bit_use_double_quant=...,
        bnb_4bit_quant_storage=...,
        **kwargs,
    ) -> None: ...
    @property
    def load_in_4bit(self):  # -> bool:
        ...
    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):  # -> None:
        ...
    @property
    def load_in_8bit(self):  # -> bool:
        ...
    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):  # -> None:
        ...
    def post_init(self):  # -> None:
        ...
    def is_quantizable(self):  # -> bool:
        ...
    def quantization_method(self):  # -> Literal['llm_int8', 'fp4', 'nf4'] | None:
        ...
    def to_dict(self) -> dict[str, Any]: ...
    def __repr__(self):  # -> str:
        ...
    def to_diff_dict(self) -> dict[str, Any]: ...

class ExllamaVersion(int, Enum):
    ONE = ...
    TWO = ...

@dataclass
class GPTQConfig(QuantizationConfigMixin):
    def __init__(
        self,
        bits: int,
        tokenizer: Any = ...,
        dataset: list[str] | str | None = ...,
        group_size: int = ...,
        damp_percent: float = ...,
        desc_act: bool = ...,
        sym: bool = ...,
        true_sequential: bool = ...,
        checkpoint_format: str = ...,
        meta: dict[str, Any] | None = ...,
        backend: str | None = ...,
        use_cuda_fp16: bool = ...,
        model_seqlen: int | None = ...,
        block_name_to_quantize: str | None = ...,
        module_name_preceding_first_block: list[str] | None = ...,
        batch_size: int = ...,
        pad_token_id: int | None = ...,
        use_exllama: bool | None = ...,
        max_input_length: int | None = ...,
        exllama_config: dict[str, Any] | None = ...,
        cache_block_outputs: bool = ...,
        modules_in_block_to_quantize: list[list[str]] | None = ...,
        **kwargs,
    ) -> None: ...
    def get_loading_attributes(self):  # -> dict[str, Any]:
        ...
    def post_init(self):  # -> None:
        ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_dict_optimum(self):  # -> dict[str, Any]:
        ...
    @classmethod
    def from_dict_optimum(cls, config_dict):  # -> Self:
        ...

@dataclass
class AwqConfig(QuantizationConfigMixin):
    def __init__(
        self,
        bits: int = ...,
        group_size: int = ...,
        zero_point: bool = ...,
        version: AWQLinearVersion = ...,
        backend: AwqBackendPackingMethod = ...,
        do_fuse: bool | None = ...,
        fuse_max_seq_len: int | None = ...,
        modules_to_fuse: dict | None = ...,
        modules_to_not_convert: list | None = ...,
        exllama_config: dict[str, int] | None = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...
    def get_loading_attributes(self):  # -> dict[str, Any]:
        ...

@dataclass
class AqlmConfig(QuantizationConfigMixin):
    def __init__(
        self,
        in_group_size: int = ...,
        out_group_size: int = ...,
        num_codebooks: int = ...,
        nbits_per_codebook: int = ...,
        linear_weights_not_to_quantize: list[str] | None = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

@dataclass
class VptqLayerConfig(QuantizationConfigMixin):
    def __init__(
        self,
        enable_norm: bool = ...,
        enable_perm: bool = ...,
        group_num: int = ...,
        group_size: int = ...,
        in_features: int = ...,
        indices_as_float: bool = ...,
        is_indice_packed: bool = ...,
        num_centroids: tuple = ...,
        num_res_centroids: tuple = ...,
        out_features: int = ...,
        outlier_size: int = ...,
        vector_lens: tuple = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

@dataclass
class VptqConfig(QuantizationConfigMixin):
    def __init__(
        self,
        enable_proxy_error: bool = ...,
        config_for_layers: dict[str, Any] = ...,
        shared_layer_config: dict[str, Any] = ...,
        modules_to_not_convert: list | None = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

@dataclass
class QuantoConfig(QuantizationConfigMixin):
    def __init__(
        self,
        weights=...,
        activations=...,
        modules_to_not_convert: list | None = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

@dataclass
class EetqConfig(QuantizationConfigMixin):
    def __init__(
        self, weights: str = ..., modules_to_not_convert: list | None = ..., **kwargs
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

class CompressedTensorsConfig(QuantizationConfigMixin):
    def __init__(
        self,
        config_groups: dict[str, QuantizationScheme | list[str]] | None = ...,
        format: str = ...,
        quantization_status: QuantizationStatus = ...,
        kv_cache_scheme: QuantizationArgs | None = ...,
        global_compression_ratio: float | None = ...,
        ignore: list[str] | None = ...,
        sparsity_config: dict[str, Any] | None = ...,
        quant_method: str = ...,
        run_compressed: bool = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...
    @classmethod
    def from_dict(
        cls, config_dict, return_unused_kwargs=..., **kwargs
    ):  # -> tuple[Self, dict[str, Any]] | Self:
        ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_diff_dict(self) -> dict[str, Any]: ...
    def get_loading_attributes(self):  # -> dict[str, bool]:
        ...
    @property
    def is_quantized(self):  # -> bool:
        ...
    @property
    def is_quantization_compressed(self):  # -> Literal[False]:
        ...
    @property
    def is_sparsification_compressed(self):  # -> Literal[False]:
        ...

@dataclass
class FbgemmFp8Config(QuantizationConfigMixin):
    def __init__(
        self,
        activation_scale_ub: float = ...,
        modules_to_not_convert: list | None = ...,
        **kwargs,
    ) -> None: ...
    def get_loading_attributes(self):  # -> dict[str, Any]:
        ...

@dataclass
class HiggsConfig(QuantizationConfigMixin):
    def __init__(
        self,
        bits: int = ...,
        p: int = ...,
        modules_to_not_convert: list[str] | None = ...,
        hadamard_size: int = ...,
        group_size: int = ...,
        tune_metadata: dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

@dataclass
class FPQuantConfig(QuantizationConfigMixin):
    def __init__(
        self,
        forward_dtype: str = ...,
        forward_method: str = ...,
        backward_dtype: str = ...,
        store_master_weights: bool = ...,
        hadamard_group_size: int | None = ...,
        pseudoquantization: bool = ...,
        transform_init: str = ...,
        modules_to_not_convert: list[str] | None = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

@dataclass
class TorchAoConfig(QuantizationConfigMixin):
    quant_method: QuantizationMethod
    quant_type: str | AOBaseConfig
    modules_to_not_convert: list | None
    quant_type_kwargs: dict[str, Any]
    include_input_output_embeddings: bool
    untie_embedding_weights: bool
    def __init__(
        self,
        quant_type: str | AOBaseConfig,
        modules_to_not_convert: list | None = ...,
        include_input_output_embeddings: bool = ...,
        untie_embedding_weights: bool = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...
    def get_apply_tensor_subclass(self):  # -> Any:
        ...
    def to_dict(self):  # -> dict[str, Any]:
        ...
    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=..., **kwargs):  # -> Self:
        ...

@dataclass
class BitNetQuantConfig(QuantizationConfigMixin):
    def __init__(
        self,
        modules_to_not_convert: list | None = ...,
        linear_class: str = ...,
        quantization_mode: str = ...,
        use_rms_norm: bool = ...,
        rms_norm_eps: float | None = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

@dataclass
class SpQRConfig(QuantizationConfigMixin):
    def __init__(
        self,
        bits: int = ...,
        beta1: int = ...,
        beta2: int = ...,
        shapes: dict[str, int] | None = ...,
        modules_to_not_convert: list[str] | None = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

@dataclass
class FineGrainedFP8Config(QuantizationConfigMixin):
    def __init__(
        self,
        activation_scheme: str = ...,
        weight_block_size: tuple[int, int] = ...,
        modules_to_not_convert: list | None = ...,
        **kwargs,
    ) -> None: ...
    def post_init(self):  # -> None:
        ...

class QuarkConfig(QuantizationConfigMixin):
    def __init__(self, **kwargs) -> None: ...

@dataclass
class Mxfp4Config(QuantizationConfigMixin):
    def __init__(
        self,
        modules_to_not_convert: list | None = ...,
        dequantize: bool = ...,
        **kwargs,
    ) -> None: ...
    def get_loading_attributes(self):  # -> dict[str, bool]:
        ...
    def to_dict(self) -> dict[str, Any]: ...
