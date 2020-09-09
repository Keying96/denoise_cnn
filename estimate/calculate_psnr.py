#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
import numpy as np
import math


def estimate_tf_psnr(ori_img, pre_img):
    ori_img = tf.image.convert_image_dtype(ori_img, tf.float32)
    pre_img = tf.image.convert_image_dtype(pre_img, tf.float32)
    psnr = tf.image.psnr(ori_img, pre_img, max_val=1.0)

    return psnr

def estimate_psnr(ori_img, pre_img, max_value):
    target = np.array(ori_img)
    ref = np.array(pre_img)

    diff = ref - target
    rmse = math.sqrt(np.mean(diff ** 2))
    psnr = 20 * math.log10(1.0/rmse)

    return psnr

# if __name__ == '__main__':
#     tf.print("hello ")