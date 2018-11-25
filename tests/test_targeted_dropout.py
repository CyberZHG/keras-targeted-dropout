import unittest
import os
import tempfile
import random
import keras
import numpy as np
from keras_targeted_dropout import TargetedDropout


class TestTargetedDropout(unittest.TestCase):

    def test_target_rate(self):
        model = keras.models.Sequential()
        model.add(TargetedDropout(input_shape=(None, None), drop_rate=0.0, target_rate=0.4))
        model.compile(optimizer='adam', loss='mse')
        model_path = os.path.join(tempfile.gettempdir(), 'keras_targeted_dropout_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={'TargetedDropout': TargetedDropout},
        )

        inputs = np.reshape(np.arange(20), (-1, 1, 4))
        outputs = model.predict(inputs)
        expected = np.array([
            [[0., 0., 0., 0.]],
            [[0., 0., 0., 0.]],
            [[8., 9., 10., 11.]],
            [[12., 13., 14., 15.]],
            [[16., 17., 18., 19.]],
        ])
        self.assertTrue(np.allclose(expected, outputs), (expected, outputs))

        inputs = np.array([
            [[1, 5, 2, 3]],
            [[4, 8, 1, 5]],
            [[5, 2, 5, 1]],
            [[2, 9, 4, 3]],
            [[4, 7, 5, 6]],
        ])
        outputs = model.predict(inputs)
        expected = np.array([
            [[0., 0., 0., 0.]],
            [[4., 8., 0., 5.]],
            [[5., 0., 5., 0.]],
            [[0., 9., 4., 0.]],
            [[4., 7., 5., 6.]],
        ])
        self.assertTrue(np.allclose(expected, outputs), (expected, outputs))

        inputs = np.random.random((100, 10, 10))
        outputs = model.predict(inputs)
        zero_num = np.sum((outputs == 0.0).astype(keras.backend.floatx()))
        actual_rate = zero_num / 10000.0
        self.assertTrue(0.39 < actual_rate < 0.41)

    def test_drop_rate(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Lambda(
            function=lambda x: TargetedDropout(
                drop_rate=0.4,
                target_rate=0.4,
            )(x, training=True),
            input_shape=(None, None, None),
        ))
        model.compile(optimizer='adam', loss='mse')
        model_path = os.path.join(tempfile.gettempdir(), 'keras_targeted_dropout_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={'TargetedDropout': TargetedDropout},
        )
        inputs = np.random.random((100, 10, 10, 10))
        outputs = model.predict(inputs)
        zero_num = np.sum((outputs == 0.0).astype(keras.backend.floatx()))
        actual_rate = zero_num / 100000.0
        self.assertTrue(0.15 < actual_rate < 0.17)
