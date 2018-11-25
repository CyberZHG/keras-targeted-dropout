import keras
import keras.backend as K


class TargetedDropout(keras.layers.Layer):
    """See: https://openreview.net/pdf?id=HkghWScuoQ"""

    def __init__(self,
                 rate,
                 **kwargs):
        """Initialize the layer.

        :param rate: Dropout rate.
        :param kwargs: Arguments for parent class.
        """
        super(TargetedDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(TargetedDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):

        def dropped_inputs():
            return None

        return K.in_train_phase(dropped_inputs, inputs, training=training)
