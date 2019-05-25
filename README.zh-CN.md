# Keras Targeted Dropout

[![Travis](https://travis-ci.org/CyberZHG/keras-targeted-dropout.svg)](https://travis-ci.org/CyberZHG/keras-targeted-dropout)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-targeted-dropout/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-targeted-dropout)
[![Version](https://img.shields.io/pypi/v/keras-targeted-dropout.svg)](https://pypi.org/project/keras-targeted-dropout/)
![Downloads](https://img.shields.io/pypi/dm/keras-targeted-dropout.svg)
![License](https://img.shields.io/pypi/l/keras-targeted-dropout.svg)

[Targeted Dropout](https://openreview.net/pdf?id=HkghWScuoQ)的非官方实现，只支持tensorflow后端。实现中没有对模型做压缩，所以不会有任何计算效率上的提升。

## 安装

```bash
pip install keras-targeted-dropout
```

## 使用

`TargetedDropout`是一个`Wrapper`，第一个参数是要被处理的原始层：

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

* `drop_rate`: 处于目标候选位置的参数的置0概率。
* `target_rate`: 选择参数作为目标候选的比例。
* `drop_patterns`: 要进行随机置0的参数名称，要求参数的名称和变量名相同，官方实现一般满足这一点，对于自定义层需要留意。
* `mode`: `TargetedDropout.MODE_UNIT`或`TargetedDropout.MODE_WEIGHT`。`MODE_UNIT`会将整列置0，`MODE_WEIGHT`会在每列中分别找元素置0。

不管那种模式，最终训练过程中被置0的概率都为`drop_rate`乘`target_rate`。
