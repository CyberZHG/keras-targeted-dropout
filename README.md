# Keras Targeted Dropout

[![Travis](https://travis-ci.org/CyberZHG/keras-targeted-dropout.svg)](https://travis-ci.org/CyberZHG/keras-targeted-dropout)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-targeted-dropout/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-targeted-dropout)

Implementation of [Targeted Dropout](https://openreview.net/pdf?id=HkghWScuoQ) with tensorflow backend.

## Install

```bash
pip install keras-targeted-dropout
```

## Usage

```python
import keras
from keras_targeted_dropout import TargetedDropout

model = keras.models.Sequential()
model.add(TargetedDropout(input_shape=(None, None), drop_rate=0.4, target_rate=0.4))
model.compile(optimizer='adam', loss='mse')
model.summary()
```

* `drop_rate`: Dropout rate for each pixel.
* `target_rate`: The proportion of bottom weights selected as candidates per channel.

The final dropout rate will be `drop_rate` times `target_rate`.

See [fashion mnist demo](./demo/mnist.py).
