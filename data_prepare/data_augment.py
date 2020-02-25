#!/usr/bin/env python
# encoding: utf-8
from imageio import  imread

import os
import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt


# train_dir = "../dataset/tmp/gt"
# batch_size = 10
# epochs = 15
# crop_size = 512
# lam_max = 20

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
    # initrialize
    images = []

    for i in range(len(imgs_list)):
        img_path = imgs_list[i]
        images.append(imread(img_path)) # 0 is grayscale mode
    # return  np.expand_dims(images, axis=0)
    # return  np.stack(images, axis=0)
    images = np.array(images).astype(np.float32)
    return  np.stack(images,axis=0)[:,:,:,None]


""" Random crops of the image """
def random_crop(images_patch, crop_size, lam_max):
    size = len(images_patch)
    # print ("size:{}".format(size))
    # print(images_patch)
    yy = np.random.randint(images_patch.shape[1] - crop_size, size=size)
    xx = np.random.randint(images_patch.shape[2]-crop_size,size=size)
    chi_rng = np.random.uniform(low=0.001, high=lam_max, size=size)
    # print ("chi_rng:{}".format(chi_rng))

    gt_patch = np.zeros((size, crop_size, crop_size, 1), dtype=images_patch.dtype)
    noise_patch = np.zeros((size, crop_size, crop_size, 1), dtype=images_patch.dtype)

    for ind in range(size):
        gt_patch[ind] = images_patch[ind, yy[ind]:yy[ind]+crop_size,xx[ind]:xx[ind]+crop_size, :]
        # noise_img = gt + mask
        noisy_mask = np.random.poisson(gt_patch[ind])
        noise_patch[ind] = noisy_mask + gt_patch[ind]

    return  noise_patch, gt_patch


""" Augment by rotating and flipping """
def augment_images_generator(noise_patch, gt_patch, batch_size):
    while True:
        inds = np.random.randint(gt_patch.shape[0], size=batch_size)
        noise_batch = np.zeros((batch_size, noise_patch.shape[1],
                                noise_patch.shape[2], 1), dtype=gt_patch.dtype)
        gt_batch = np.zeros((batch_size, gt_patch.shape[1],
                             gt_patch.shape[2], 1), dtype=gt_patch.dtype)

        for i,ind in enumerate(inds):
            # print("ind:{}".format(ind))
            # write_noiseimg(gt_patch[ind], "../dataset/tmp/train", "ori_{}.png".format(ind))
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

            # if i == 0:
            #     print ("this is 0")
            #     write_noiseimg(noise_batch[i], "../dataset/tmp/train", "noise_{}.png".format(ind))
            #     write_noiseimg(gt_batch[i], "../dataset/tmp/train", "gt_{}.png".format(ind))

            # return noise_batch, gt_batch
            yield noise_batch, gt_batch




# train_images = load_images(train_dir)
# print(train_images.shape)
#
# images_patch = train_images[:-5]
# X_val = train_images[-5:]
# print('%d training images'%len(images_patch))
# print('%d validation images'%len(X_val))
#
# noise_patch, gt_patch = random_crop(images_patch, crop_size, lam_max)
# augment_images_generator(noise_patch, gt_patch, batch_size)

