from xlstm.xlstm_large.model import (
    BackendModeType,
    ChunkwiseKernelType,
    DtypeType,
    SequenceKernelType,
    StepKernelType,
    WeightModeType,
)

from ...configuration_utils import PretrainedConfig
from ...utils import is_xlstm_available

"""xLSTM configuration."""
if is_xlstm_available():
    external_xlstm = ...
else:
    BackendModeType = ...
    ChunkwiseKernelType = ...
    DtypeType = ...
    SequenceKernelType = ...
    StepKernelType = ...
    WeightModeType = ...
    def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int: ...

    external_xlstm = ...
logger = ...

class xLSTMConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size: int = ...,
        hidden_size: int = ...,
        embedding_dim: int | None = ...,
        num_hidden_layers: int | None = ...,
        num_blocks: int | None = ...,
        num_heads: int = ...,
        use_bias: bool = ...,
        norm_reduction_force_float32: bool = ...,
        tie_word_embeddings: bool = ...,
        add_out_norm: bool = ...,
        norm_eps: float = ...,
        qk_dim_factor: float = ...,
        v_dim_factor: float = ...,
        chunkwise_kernel: ChunkwiseKernelType = ...,
        sequence_kernel: SequenceKernelType = ...,
        step_kernel: StepKernelType = ...,
        mode: BackendModeType = ...,
        chunk_size: int = ...,
        return_last_states: bool = ...,
        autocast_kernel_dtype: DtypeType = ...,
        eps: float = ...,
        inference_state_dtype: DtypeType = ...,
        ffn_proj_factor: float = ...,
        ffn_round_up_to_multiple_of: int = ...,
        gate_soft_cap: float = ...,
        output_logit_soft_cap: float = ...,
        weight_mode: WeightModeType = ...,
        use_cache: bool = ...,
        pad_token_id: int = ...,
        bos_token_id: int = ...,
        eos_token_id: int = ...,
        max_inference_chunksize: int = ...,
        **kwargs,
    ) -> None: ...
    @property
    def qk_dim(self):  # -> int:
        ...
    @property
    def v_dim(self):  # -> int:
        ...
    @property
    def qk_head_dim(self):  # -> int:
        ...
    @property
    def v_head_dim(self):  # -> int:
        ...
    def to_xlstm_block_config(self):  # -> Self:
        ...

__all__ = ["xLSTMConfig"]
