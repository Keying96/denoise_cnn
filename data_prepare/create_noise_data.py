#!/usr/bin/env python
# encoding: utf-8
import  numpy as np
import os

from data_prepare.data_generate import *
from data_prepare.tools import *


# input_dir = "../dataset/super_resolution/sp_test/test"
# output_dir = os.path.join(input_dir, "output_tiles/random_tiles")
input_dir = "../dataset/caltechPedestrians/parallel_test"
output_dir = os.path.join(input_dir, "noise_img/")


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

def poisson_noise_imgs(gt_set, lam):
    """
    add noise on rgb images
    :param x: rgb array
    :return:  noise rgb
    """
    noise_set = []

    for x in gt_set:
        range = np.sqrt(np.multiply(x, lam))
        a = range / 2
        chi_rng = np.random.uniform(low= -a, high= a,size= x.shape)

        noise_img = chi_rng + x
        ouput = np.minimum(np.maximum(noise_img, 0), 255)
        noise_set.append(ouput)

    return noise_set

if __name__ == '__main__':
    lam_noise = 20
    gt_imgs, image_names = load_data_images(input_dir)
    noise_set = poisson_noise_imgs(gt_imgs, lam_noise)

    for ind, test_img in enumerate(noise_set):
        noise_name = image_names[ind].split(".") [0] + "_noise.jpg"
        write_img(noise_set[ind], output_dir,noise_name)

