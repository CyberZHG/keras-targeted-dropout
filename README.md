# Keras Targeted Dropout

[![Travis](https://travis-ci.org/CyberZHG/keras-targeted-dropout.svg)](https://travis-ci.org/CyberZHG/keras-targeted-dropout)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-targeted-dropout/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-targeted-dropout)
[![Version](https://img.shields.io/pypi/v/keras-targeted-dropout.svg)](https://pypi.org/project/keras-targeted-dropout/)
![Downloads](https://img.shields.io/pypi/dm/keras-targeted-dropout.svg)
![License](https://img.shields.io/pypi/l/keras-targeted-dropout.svg)

Unofficial implementation of [Targeted Dropout](https://openreview.net/pdf?id=HkghWScuoQ) with tensorflow backend.
Note that there is no model compression in this implementation.

## Install

```bash
pip install keras-targeted-dropout
```

## Usage

```python
import keras
from keras_targeted_dropout import TargetedDropout

model = keras.models.Sequential()
model.add(TargetedDropout(
    layer=keras.layers.Dense(units=2, activation='softmax'),
    drop_rate=0.8,
    target_rate=0.2,
    drop_patterns=['kernel'],
    mode=TargetedDropout.MODE_UNIT,
    input_shape=(5,),
))
model.compile(optimizer='adam', loss='mse')
model.summary()
```

* `drop_rate`: Dropout rate for each pixel.
* `target_rate`: The proportion of bottom weights selected as candidates
* `drop_patterns`: A list of names of weights to be dropped.
* `mode`: `TargetedDropout.MODE_UNIT` or `TargetedDropout.MODE_WEIGHT`.

The final dropout rate will be `drop_rate` times `target_rate`.
