from ...configuration_utils import PretrainedConfig

"""VITS model configuration"""
logger = ...

class VitsConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        window_size=...,
        use_bias=...,
        ffn_dim=...,
        layerdrop=...,
        ffn_kernel_size=...,
        flow_size=...,
        spectrogram_bins=...,
        hidden_act=...,
        hidden_dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_stochastic_duration_prediction=...,
        num_speakers=...,
        speaker_embedding_size=...,
        upsample_initial_channel=...,
        upsample_rates=...,
        upsample_kernel_sizes=...,
        resblock_kernel_sizes=...,
        resblock_dilation_sizes=...,
        leaky_relu_slope=...,
        depth_separable_channels=...,
        depth_separable_num_layers=...,
        duration_predictor_flow_bins=...,
        duration_predictor_tail_bound=...,
        duration_predictor_kernel_size=...,
        duration_predictor_dropout=...,
        duration_predictor_num_flows=...,
        duration_predictor_filter_channels=...,
        prior_encoder_num_flows=...,
        prior_encoder_num_wavenet_layers=...,
        posterior_encoder_num_wavenet_layers=...,
        wavenet_kernel_size=...,
        wavenet_dilation_rate=...,
        wavenet_dropout=...,
        speaking_rate=...,
        noise_scale=...,
        noise_scale_duration=...,
        sampling_rate=...,
        **kwargs,
    ) -> None: ...

__all__ = ["VitsConfig"]
