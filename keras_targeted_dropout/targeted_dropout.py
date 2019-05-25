import tensorflow as tf
from .backend import keras
from .backend import backend as K


class TargetedDropout(keras.layers.Wrapper):
    """See: https://openreview.net/pdf?id=HkghWScuoQ"""

    MODE_UNIT = 'unit'
    MODE_WEIGHT = 'weight'

    def __init__(self,
                 layer,
                 drop_rate,
                 target_rate,
                 drop_patterns,
                 mode=MODE_WEIGHT,
                 **kwargs):
        """Initialize the layer.

        :param drop_rate: Dropout rate.
        :param target_rate: Targeting proportion.
        :param drop_patterns: A list of names of weights to be dropped.
        :param mode: Dropout mode, 'weight' or 'unit'.
        :param kwargs: Arguments for parent class.
        """
        super(TargetedDropout, self).__init__(layer, **kwargs)
        self.supports_masking = True
        self.drop_rate = drop_rate
        self.target_rate = target_rate
        self.drop_patterns = drop_patterns
        self.mode = mode

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
        super(TargetedDropout, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def _in_drop_patterns(self, weight):
        return any(p in weight.name for p in self.drop_patterns)

    def call(self, inputs, mask=None, training=None):
        kwargs = {}
        if keras.utils.generic_utils.has_arg(self.layer.call, 'mask'):
            kwargs['mask'] = mask
        if keras.utils.generic_utils.has_arg(self.layer.call, 'training'):
            kwargs['training'] = training

        origins = {}
        for i, w in enumerate(self.layer._trainable_weights):
            name = w.name.split('/')[-1].split(':')[0]
            if self._in_drop_patterns(w) and hasattr(self.layer, name):
                origins[name] = getattr(self.layer, name)
                setattr(self.layer, name, self._drop_weights(self.layer._trainable_weights[i], training))

        outputs = self.layer.call(inputs, **kwargs)

        for name, attr in origins.items():
            setattr(self.layer, name, attr)
        return outputs

    def _drop_weights(self, w, training):
        if self.mode == self.MODE_WEIGHT:
            target_mask = self._compute_weight_mask(w)
        else:
            target_mask = self._compute_unit_mask(w)

        def dropped_mask():
            drop_mask = K.cast(K.random_uniform(K.shape(w)) < self.drop_rate, K.floatx())
            return target_mask * drop_mask

        def pruned_mask():
            return target_mask

        mask = K.in_train_phase(dropped_mask, pruned_mask, training=training)

        outputs = (1.0 - mask) * w
        return outputs

    def _compute_weight_mask(self, w):
        input_shape, input_type = K.int_shape(w), K.dtype(w)
        reshaped = K.abs(K.reshape(w, (-1, input_shape[-1])))
        index = int(K.get_value(K.minimum(
            round(self.target_rate * K.int_shape(reshaped)[0]),
            K.int_shape(reshaped)[0] - 1,
        )))
        threshold = tf.sort(reshaped, axis=0)[index:index + 1]
        target_mask = K.reshape(K.cast(reshaped < threshold, dtype=input_type), input_shape)
        return target_mask

    def _compute_unit_mask(self, w):
        input_shape, input_type = K.int_shape(w), K.dtype(w)
        reshaped = K.reshape(w, (-1, input_shape[-1]))
        norm = K.sqrt(K.sum(K.square(reshaped), axis=0) + K.epsilon())
        index = int(K.get_value(K.minimum(
            round(self.target_rate * K.int_shape(norm)[0]),
            K.int_shape(norm)[0] - 1,
        )))
        threshold = tf.sort(norm)[index]
        target_mask = K.cast(norm < threshold, dtype=input_type)
        while K.ndim(target_mask) < K.ndim(w):
            target_mask = K.expand_dims(target_mask, axis=0)
        return target_mask

    def get_config(self):
        config = {
            'drop_rate': self.drop_rate,
            'target_rate': self.target_rate,
            'drop_patterns': self.drop_patterns,
            'mode': self.mode,
        }
        base_config = super(TargetedDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
