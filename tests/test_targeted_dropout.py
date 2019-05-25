import unittest
import os
import tempfile
import numpy as np
from keras_targeted_dropout.backend import keras
from keras_targeted_dropout.backend import backend as K
from keras_targeted_dropout import TargetedDropout


class Weight(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Weight, self).__init__(**kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(1000, 100),
            name='kernel',
            initializer='glorot_uniform',
        )

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.kernel)

    def call(self, inputs, mask=None, training=None):
        return K.expand_dims(self.kernel, axis=0)


class TestTargetedDropout(unittest.TestCase):

    def test_weight_drop_shape(self):
        model = keras.models.Sequential()
        model.add(TargetedDropout(
            layer=Weight(),
            drop_rate=0.0,
            target_rate=0.35,
            drop_patterns=['kernel'],
            mode=TargetedDropout.MODE_WEIGHT,
            input_shape=(1,)
        ))
        model.compile('sgd', 'mse')

        model_path = os.path.join(tempfile.gettempdir(), 'keras_targeted_dropout_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'Weight': Weight,
            'TargetedDropout': TargetedDropout,
        })

        dropped = model.predict(np.ones((1, 1)))[0]
        row, col = dropped.shape
        for c in range(col):
            rate = np.sum(np.equal(dropped[:, c], 0.0)) / row
            self.assertGreater(rate, 0.34)
            self.assertLess(rate, 0.36)
        rate = np.mean(np.equal(dropped, 0.0))
        self.assertGreater(rate, 0.34)
        self.assertLess(rate, 0.36)

    def test_unit_drop_shape(self):
        model = keras.models.Sequential()
        model.add(TargetedDropout(
            layer=Weight(),
            drop_rate=0.0,
            target_rate=0.35,
            drop_patterns=['kernel'],
            mode=TargetedDropout.MODE_UNIT,
            input_shape=(1,)
        ))
        model.compile('sgd', 'mse')

        model_path = os.path.join(tempfile.gettempdir(), 'keras_targeted_dropout_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'Weight': Weight,
            'TargetedDropout': TargetedDropout,
        })

        dropped = model.predict(np.ones((1, 1)))[0]
        col = dropped.shape[-1]
        count = 0
        for c in range(col):
            if np.sum(dropped[:, c]) == 0.0:
                count += 1
        self.assertEqual(35, count)
        rate = np.mean(np.equal(dropped, 0.0))
        self.assertGreater(rate, 0.34)
        self.assertLess(rate, 0.36)

    @staticmethod
    def _get_data(w=None):
        if w is None:
            w = np.random.standard_normal((5, 2))
        x = np.random.standard_normal((2048, 5))
        y = np.dot(x, w).argmax(axis=-1)
        return x, y, w

    def test_fit(self):
        model = keras.models.Sequential()
        model.add(TargetedDropout(
            layer=keras.layers.Dense(units=3, activation='tanh'),
            drop_rate=0.8,
            target_rate=0.2,
            drop_patterns=['kernel'],
            mode=TargetedDropout.MODE_WEIGHT,
            input_shape=(5,)
        ))
        model.add(TargetedDropout(
            layer=keras.layers.Dense(units=2, activation='softmax'),
            drop_rate=0.8,
            target_rate=0.2,
            drop_patterns=['kernel'],
            mode=TargetedDropout.MODE_UNIT,
        ))
        model.compile('adam', 'sparse_categorical_crossentropy')

        x, y, w = self._get_data()
        model.fit(x, y, epochs=50)

        model_path = os.path.join(tempfile.gettempdir(), 'keras_targeted_dropout_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'TargetedDropout': TargetedDropout})

        x, y, _ = self._get_data(w)
        predicted = model.predict(x).argmax(axis=-1)
        self.assertLess(np.sum(np.not_equal(y, predicted)), x.shape[0] // 10)
