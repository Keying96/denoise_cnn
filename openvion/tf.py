#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import cv2

print(tf.__version__)

imag= "/home/zhu/PycharmProjects/denoise_cnn/dataset/caltechPedestrians" \
      "/parallel_test/noise_img/94079_noise.jpg"

image = cv2.imread(imag)
# cv2.imshow("show",imag)
# cv2.waitKey(0)
# cv2.destroyAllWindows()