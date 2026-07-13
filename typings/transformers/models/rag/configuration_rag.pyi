from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings

"""RAG model configuration"""
RAG_CONFIG_DOC = ...

@add_start_docstrings(RAG_CONFIG_DOC)
class RagConfig(PretrainedConfig):
    model_type = ...
    has_no_defaults_at_init = ...
    def __init__(
        self,
        vocab_size=...,
        is_encoder_decoder=...,
        prefix=...,
        bos_token_id=...,
        pad_token_id=...,
        eos_token_id=...,
        decoder_start_token_id=...,
        title_sep=...,
        doc_sep=...,
        n_docs=...,
        max_combined_length=...,
        retrieval_vector_size=...,
        retrieval_batch_size=...,
        dataset=...,
        dataset_split=...,
        index_name=...,
        index_path=...,
        passages_path=...,
        use_dummy_dataset=...,
        reduce_loss=...,
        label_smoothing=...,
        do_deduplication=...,
        exclude_bos_score=...,
        do_marginalize=...,
        output_retrieved=...,
        use_cache=...,
        forced_eos_token_id=...,
        dataset_revision=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_question_encoder_generator_configs(
        cls,
        question_encoder_config: PretrainedConfig,
        generator_config: PretrainedConfig,
        **kwargs,
    ) -> PretrainedConfig: ...

__all__ = ["RagConfig"]
