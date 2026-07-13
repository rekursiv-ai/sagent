import os

from ....configuration_utils import PretrainedConfig

"""Jukebox configuration"""
logger = ...
_LARGE_ATTENTION = ...
_RawColumnPreviousRowAttention = ...
_FullDenseAttention = ...
_PrimePrimeDenseAttention = ...

def full_dense_attention(layer):  # -> str:
    ...
def raw_column_previous_row_attention(layer): ...
def large_separated_enc_dec_w_lyrics(layer): ...
def enc_dec_with_lyrics(layer): ...

ATTENTION_PATTERNS = ...

class JukeboxPriorConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        act_fn=...,
        level=...,
        alignment_head=...,
        alignment_layer=...,
        attention_multiplier=...,
        attention_pattern=...,
        attn_dropout=...,
        attn_res_scale=...,
        blocks=...,
        conv_res_scale=...,
        num_layers=...,
        emb_dropout=...,
        encoder_config=...,
        encoder_loss_fraction=...,
        hidden_size=...,
        init_scale=...,
        is_encoder_decoder=...,
        lyric_vocab_size=...,
        mask=...,
        max_duration=...,
        max_nb_genres=...,
        merged_decoder=...,
        metadata_conditioning=...,
        metadata_dims=...,
        min_duration=...,
        mlp_multiplier=...,
        music_vocab_size=...,
        n_ctx=...,
        n_heads=...,
        nb_relevant_lyric_tokens=...,
        res_conv_depth=...,
        res_conv_width=...,
        res_convolution_multiplier=...,
        res_dilation_cycle=...,
        res_dilation_growth_rate=...,
        res_downs_t=...,
        res_strides_t=...,
        resid_dropout=...,
        sampling_rate=...,
        spread=...,
        timing_dims=...,
        zero_out=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, level=..., **kwargs
    ):  # -> Self:
        ...

class JukeboxVQVAEConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        act_fn=...,
        nb_discrete_codes=...,
        commit=...,
        conv_input_shape=...,
        conv_res_scale=...,
        embed_dim=...,
        hop_fraction=...,
        levels=...,
        lmu=...,
        multipliers=...,
        res_conv_depth=...,
        res_conv_width=...,
        res_convolution_multiplier=...,
        res_dilation_cycle=...,
        res_dilation_growth_rate=...,
        res_downs_t=...,
        res_strides_t=...,
        sample_length=...,
        init_scale=...,
        zero_out=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ):  # -> Self:
        ...

class JukeboxConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vqvae_config=...,
        prior_config_list=...,
        nb_priors=...,
        sampling_rate=...,
        timing_dims=...,
        min_duration=...,
        max_duration=...,
        max_nb_genres=...,
        metadata_conditioning=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_configs(
        cls,
        prior_configs: list[JukeboxPriorConfig],
        vqvae_config: JukeboxVQVAEConfig,
        **kwargs,
    ):  # -> Self:
        ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

__all__ = ["JukeboxConfig", "JukeboxPriorConfig", "JukeboxVQVAEConfig"]
