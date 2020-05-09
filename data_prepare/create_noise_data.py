#!/usr/bin/env python
# encoding: utf-8
import  numpy as np

def add_train_poisson_noise(x, lam):
    """
    add noise on rgb images
    :param x: rgb array
    :return:  noise rgb
    """
    range = np.sqrt(np.multiply(x, lam))
    a = range / 2
    chi_rng = np.random.uniform(low= -a, high= a,size= x.shape)

    noise_img = chi_rng + x
    ouput = np.minimum(np.maximum(noise_img, 0), 255)
    return ouput

