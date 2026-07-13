from ....modeling_tf_utils import keras

"""
A TF 2.0 Adaptive Softmax for Transformer XL model.
"""

class TFAdaptiveSoftmaxMask(keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        d_embed,
        d_proj,
        cutoffs,
        div_val=...,
        keep_order=...,
        **kwargs,
    ) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, hidden, target, return_mean=..., training=...): ...
