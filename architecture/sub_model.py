#!/usr/bin/env python
# encoding: utf-8

from cnn.model_5layers import *

weights_path = "../checkpoint_list/checkpoint_UNet_5layers_200612_3092/UNet_5layers_200612_30902"
ckpt_path = "../checkpoint_list/checkpoint_UNet_5layers_200612_3092/"
input_dir = "../dataset/caltechPedestrians/parallel_test"

# 自定义激活函数
def lrelu(x):
    return tf.maximum(x * 0.2, x)


def encoder_input(unet_model, input_size=(512, 512, 3)):

    input = unet_model.input.shape
    input_layer = unet_model.input
    encoder_conv1_1 = unet_model.layers[1]
    encoder_conv1_2 = unet_model.layers[2]
    encoder_input_model = tf.keras.Sequential([tf.keras.Input(shape=(input[1], input[2],input[3])),
                         encoder_conv1_1,
                         encoder_conv1_2])
    return encoder_input_model


def encoder_conv2(unet_model, input_size=(256, 256, 32)):
    conv2_input  = unet_model.layers[3].input.shape
    encoder_pool = unet_model.layers[3]
    encoder_conv2_1  = unet_model.layers[4]
    encoder_conv2_2 = unet_model.layers[5]
    encoder_conv2_model = tf.keras.Sequential([tf.keras.Input(shape=(conv2_input[1],conv2_input[2],conv2_input[3])),
                                               encoder_pool,
                                               encoder_conv2_1,
                                               encoder_conv2_2])
    return encoder_conv2_model

def encoder_conv3(unet_model, input_size=(128, 128, 64)):
    conv3_input = unet_model.layers[6].input.shape
    encoder_conv2_pool = unet_model.layers[6]
    encoder_conv3_1  = unet_model.layers[7]
    encoder_conv3_2 = unet_model.layers[8]
    encoder_conv3_model = tf.keras.Sequential([tf.keras.Input(shape=(conv3_input[1],conv3_input[2],conv3_input[3])),
                                               encoder_conv2_pool,
                                               encoder_conv3_1,
                                               encoder_conv3_2])
    return encoder_conv3_model

def encoder_conv8(unet_model, a=(128,128,128), b=(256,256,64)):
    a_shape = unet_model.layers[9].input.shape
    b_shape = unet_model.layers[6].input.shape

    a = tf.keras.Input(shape=(a_shape[1], a_shape[2], a_shape[3]))
    b = tf.keras.Input(shape=(b_shape[1], b_shape[2], b_shape[3]))

    upsample_concat1 = unet_model.layers[9](a,b)
    decoder_conv8_1 = unet_model.layers[10](upsample_concat1)
    decoder_conv8_2 = unet_model.layers[11](decoder_conv8_1)
    decoder_conv8_model = tf.keras.Model([a,b], decoder_conv8_2)

    return decoder_conv8_model


def encoder_output(unet_model, a=(256,256,64), b=(512,512,32)):
    a_shape = unet_model.layers[12].input.shape
    b_shape = unet_model.layers[3].input.shape

    a = tf.keras.Input(shape=(a_shape[1], a_shape[2], a_shape[3]))
    b = tf.keras.Input(shape=(b_shape[1], b_shape[2], b_shape[3]))

    upsample_concat2 = unet_model.layers[12](a,b)
    decoder_conv9_1 = unet_model.layers[13](upsample_concat2)
    decoder_conv9_2 = unet_model.layers[14](decoder_conv9_1)
    output = unet_model.layers[15](decoder_conv9_2)
    output_model = tf.keras.Model([a,b], output)

    return output_model

def load_sub_models(weights_path, input_size = (321, 481, 3)):
    from cnn.model_5layers import unet
    unet_model, unet_name = unet(weights_path, input_size)
    sub_models = []

    input_model = encoder_input(unet_model)
    sub_models.append(input_model)

    encoder_conv2_model = encoder_conv2(unet_model)
    sub_models.append(encoder_conv2_model)

    encoder_conv3_model = encoder_conv3(unet_model)
    sub_models.append(encoder_conv3_model)

    decoder_conv8_model = encoder_conv8(unet_model)
    sub_models.append(decoder_conv8_model)

    output_model = encoder_output(unet_model)
    sub_models.append(output_model)

    return sub_models


if __name__ == '__main__':
    load_sub_models(weights_path)
