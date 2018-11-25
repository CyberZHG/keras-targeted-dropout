import keras
import keras.backend as K


class TargetedDropout(keras.layers.Layer):
    """See: https://openreview.net/pdf?id=HkghWScuoQ"""

    def __init__(self,
                 drop_rate,
                 target_rate,
                 **kwargs):
        """Initialize the layer.

        :param drop_rate: Dropout rate.
        :param target_rate: Targeting proportion.
        :param kwargs: Arguments for parent class.
        """
        super(TargetedDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.drop_rate = drop_rate
        self.target_rate = target_rate

    def get_config(self):
        config = {
            'drop_rate': self.drop_rate,
            'target_rate': self.target_rate,
        }
        base_config = super(TargetedDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _compute_target_mask(self, inputs):
        input_shape = K.shape(inputs)
        channel_dim = K.prod(input_shape[:-1])
        norm = K.abs(inputs)
        channeled_norm = K.transpose(K.reshape(norm, (channel_dim, input_shape[-1])))
        idx = K.cast(self.target_rate * K.cast(channel_dim, K.floatx()), 'int32')
        threshold = -K.tf.nn.top_k(-channeled_norm, k=idx).values[:, -1]
        threshold = K.reshape(K.tile(threshold, [channel_dim]), input_shape)
        target_mask = K.switch(
            norm <= threshold,
            K.ones_like(inputs, dtype=K.floatx()),
            K.zeros_like(inputs, dtype=K.floatx()),
        )
        return target_mask

    def call(self, inputs, training=None):
        target_mask = self._compute_target_mask(inputs)

        def dropped_mask():
            drop_mask = K.switch(
                K.random_uniform(K.shape(inputs)) < self.drop_rate,
                K.ones_like(inputs, K.floatx()),
                K.zeros_like(inputs, K.floatx()),
            )
            return target_mask * drop_mask

        def pruned_mask():
            return target_mask

        mask = K.in_train_phase(dropped_mask, pruned_mask, training=training)
        outputs = K.switch(
            mask > 0.5,
            K.zeros_like(inputs, dtype=K.dtype(inputs)),
            inputs,
        )
        return outputs
