#!/usr/bin/env python
# encoding: utf-8
from cnn.UpSampleConcat import *

import tensorflow as tf

# custom activation function
def lrelu(x):
    return tf.maximum(x * 0.2, x)

def unet(pretrained_weight = None, input_size = (512, 512, 3)):
    #define input layer
    input = layers.Input(input_size)

    #block 1
    conv1 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv1_1')(input)
    conv1 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv1_2')(conv1)
    pool1 = layers.MaxPool2D(pool_size=(2, 2), padding='SAME')(conv1)

    #block 2
    conv2 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv2_1')(pool1)
    conv2 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv2_2')(conv2)
    pool2 = layers.MaxPool2D([2, 2], padding='SAME')(conv2)

    #block 3
    conv3 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv3_1')(pool2)
    conv3 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv3_2')(conv3)
    pool3 = layers.MaxPool2D([2, 2], padding='SAME')(conv3)

    #block 4
    conv4 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv4_1')(pool3)
    conv4 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv4_2')(conv4)

    #block 7
    up7 = UpSampleConcat(128, 256,name = "Variable_1")(conv4,conv3)
    conv7 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv7_1')(up7)
    conv7 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv7_2')(conv7)

    #block 8
    up8 = UpSampleConcat(64, 128,name = "Variable_2")(conv7,conv2)
    conv8 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv8_1')(up8)
    conv8 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv8_2')(conv8)

    #block 9
    up9 = UpSampleConcat(32, 64,name = "Variable_3")(conv8,conv1)
    conv9 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv9_1')(up9)
    conv9 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv9_2')(conv9)

    #block 10
    output = layers.Conv2D(3, [1, 1], padding='SAME', activation=None, name='g_conv10')(conv9)

    model = tf.keras.Model(inputs = input, outputs = output)


    if (pretrained_weight):
        model.load_weights(pretrained_weight)

    return  model,"UNet_7layers"
