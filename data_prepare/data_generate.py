#!/usr/bin/env python
# encoding: utf-8
from imageio import  imread
from data_prepare.create_noise_data import *

import os
import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt


def write_noiseimg(poisson_img, output_dir, input_name):
    output_dir = os.path.join(output_dir, input_name)
    imageio.imwrite(output_dir, poisson_img.astype("uint8"))

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img, cmap = "gray")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

""" Load dataset"""
def load_images(data_dir):
    imgs_list = glob.glob(data_dir + "/*.png") # get name list of all .png files
    print ("the size of imgs: {}".format(len(imgs_list)))

    # initrialize
    images = []

    for i in range(len(imgs_list)):
        img_path = imgs_list[i]
        images.append(imread(img_path)) # 0 is grayscale mode

    images = np.array(images).astype(np.float32)
    # return  np.stack(images,axis=0)[:,:,:,None]
    return  np.stack(images,axis=0)


""" Random crops of the image """
def random_crop(images_patch, crop_size, lam_noise):
    size = len(images_patch)

    yy = np.random.randint(images_patch.shape[1] - crop_size, size=size)
    xx = np.random.randint(images_patch.shape[2]-crop_size,size=size)

    crop_path = np.zeros((size, crop_size, crop_size, 3), dtype=images_patch.dtype)
    gt_patch = np.zeros((size, crop_size, crop_size, 3), dtype=images_patch.dtype)
    noise_patch = np.zeros((size, crop_size, crop_size, 3 ), dtype=images_patch.dtype)

    for ind in range(size):
        crop_path[ind] = images_patch[ind, yy[ind]:yy[ind]+crop_size,xx[ind]:xx[ind]+crop_size, :]
        curr_gt = crop_path[ind]
        gt_patch[ind] = curr_gt / 255.0
        noise_patch[ind] = add_train_poisson_noise(curr_gt, lam_noise) / 255.0

        # write_noiseimg(noise_patch[ind], test_save_dir, "noise_{}.png".format(ind))
        # write_noiseimg(gt_patch[ind], test_save_dir, "gt_{}.png".format(ind))
        # print (" crop and noise")

    return  noise_patch, gt_patch


""" Augment by rotating and flipping """
def augment_images_generator(noise_patch, gt_patch, batch_size):
    while True:
        inds = np.random.randint(gt_patch.shape[0], size=batch_size)
        noise_batch = np.zeros((batch_size, noise_patch.shape[1],
                                noise_patch.shape[2], gt_patch.shape[3]), dtype=gt_patch.dtype)
        gt_batch = np.zeros((batch_size, gt_patch.shape[1],
                             gt_patch.shape[2], gt_patch.shape[3]), dtype=gt_patch.dtype)

        for i,ind in enumerate(inds):
            # write_noiseimg(gt_patch[ind], test_save_dir, "ori_{}.png".format(ind))
            if np.random.randint(2, size=1)[0] == 1:  # random flip
                # print("v")
                noise_batch[i] = np.flip(noise_patch[ind], axis=0)
                gt_batch[i] = np.flip(gt_patch[ind], axis=0)
            if np.random.randint(2, size=1)[0] == 1:
                # print("h")
                noise_batch[i] = np.flip(noise_patch[ind], axis=1)
                gt_batch[i] = np.flip(gt_patch[ind], axis=1)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                # print ("tran")
                noise_batch[i] = np.transpose(noise_patch[ind], (1, 0, 2))
                gt_batch[i] = np.transpose(gt_patch[ind], (1, 0, 2))

            # write_noiseimg(noise_batch[i], test_save_dir, "noise_{}.png".format(ind))
            # write_noiseimg(gt_batch[i], test_save_dir, "gt_{}.png".format(ind))

            # return noise_batch, gt_batch
            yield noise_batch, gt_batch

# train_images = load_images(rgb_dir)
# print(train_images.shape)
#
# images_patch = train_images[:-1]
# X_val = train_images[-1:]
# print('%d training images'%len(images_patch))
# print('%d validation images'%len(X_val))

# noise_patch, gt_patch = random_crop(images_patch, crop_size, lam_noise)
# augment_images_generator(noise_patch, gt_patch, batch_size)
