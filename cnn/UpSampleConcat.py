#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class UpSampleConcat(layers.Layer):
    def __init__(self,output_channels, in_channels, **kwargs):
        # 参数**kwargs代表按字典方式继承父类
        self.pool_size = 2
        self.output_channels = output_channels
        self.in_channels = in_channels
        self.W = tf.Variable(tf.random.truncated_normal([2, 2, output_channels, in_channels],
                                                        stddev=0.02), name = "w1")
        # self.W = tf.keras.backend.variable(tf.random.truncated_normal([2, 2, output_channels, in_channels],
        #                                                 stddev=0.02), name = "w1")
        super(UpSampleConcat, self).__init__(**kwargs)

    # def build(self, input_shape):
        # self.W = tf.Variable(tf.random.truncated_normal([2, 2, self.output_channels, self.in_channels],
        #                                                 stddev=0.02), name = "w1")

    def call(self, x1,x2):
        """
        :param x1: conv5
        :param x2: conv4
        :return:
        """
        deconv = tf.nn.conv2d_transpose(x1, self.W, tf.shape(x2),
                                        strides=[1, self.pool_size, self.pool_size, 1])
        # deconv = layers.Lambda(lambda x: tf.nn.conv2d_transpose(x, self.W, tf.shape(x2),
        #                                 strides=[1, self.pool_size, self.pool_size, 1]))(x1)
        deconv_output = layers.concatenate([deconv,x2], axis = 3)
        deconv_output.set_shape([None, None, None, self.output_channels * 2])
        # deconv_output = layers.Lambda(lambda x:x.set_shape([None, None, None, self.output_channels * 2]))(deconv_output)

        return deconv_output

    # custom layer to be serializable as part of a functional cnn_model
    def get_config(self):
        config = super(UpSampleConcat,self).get_config()
        config.update({"output_channels": self.output_channels,
                      "in_channels": self.in_channels,
                       "up_sample_pool_size":self.pool_size})
        #
        # config.update({"output_channels": self.output_channels,
        #               "in_channels": self.in_channels})

        return config