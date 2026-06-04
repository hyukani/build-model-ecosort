import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    register_keras_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_keras_serializable = keras.utils.register_keras_serializable


@register_keras_serializable(package='EcoSort', name='ChannelAttention')
class ChannelAttention(layers.Layer):
    def __init__(self, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden_units = max(channels // self.reduction, 8)
        self.dense_1 = layers.Dense(hidden_units, activation='relu')
        self.dense_2 = layers.Dense(channels, activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        attention = self.global_pool(inputs)
        attention = self.dense_1(attention)
        attention = self.dense_2(attention)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({'reduction': self.reduction})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
