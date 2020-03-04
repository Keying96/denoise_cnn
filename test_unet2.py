#!/usr/bin/env python
# encoding: utf-8
from model import *
# from  model2 import  *
from imageio import imread
from data_prepare.create_noise_data import  *

import  numpy as  np
import  tensorflow as tf
import os
import imageio
import glob

weights_path = "./dataset/checkpoint/"
test_dir = "./dataset/Sony/test"
output_dir = "./dataset/Sony/output_20200304/"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

lam_noise = 20

def write_noiseimg(poisson_img, output_dir, input_name):
    output_dir = os.path.join(output_dir, input_name)
    imageio.imwrite(output_dir, poisson_img.astype("uint8"))

""" Load dataset"""
def load_images(data_dir):
    imgs_list = glob.glob(data_dir + "/*.png")  # get name list of all .png files
    images = []
    imgs_name = []

    for i in range(len(imgs_list)):
        img_path = imgs_list[i]
        ori_img = imread(img_path)

        img_noise = add_train_poisson_noise(ori_img, lam_noise) / 255.0
        images.append(img_noise) # 0 is grayscale mode

        name = os.path.basename(imgs_list[i])
        imgs_name.append(name)
        # img_name = os.path.basename(name.split(".png")[0] + "_noise.png"
        # write_noiseimg(img_noise * 255.0, output_dir, imgs_name)

    images = np.array(images).astype(np.float32)
    # return  np.stack(images,axis=0)[:,:,:,None], imgs_name
    return  np.stack(images,axis=0), imgs_name

""" Denoise test images"""
test_set, imgs_name = load_images(test_dir)

""" Re-create the model and load the weights """
H = test_set.shape[1]
W = test_set.shape[2]
C = test_set.shape[3]

model = unet(input_size=(H, W, C))
model.load_weights(weights_path)
model.summary()

for ind, img in enumerate(test_set):
    test_img = np.expand_dims(img, axis=0)

    # gt_img = img
    print ("gt_img.shape:{}".format(test_img.shape))
    img_name = os.path.basename(imgs_name[ind]).split(".png")[0] + "_test.png"
    write_noiseimg(test_img[0] * 255.0 , output_dir, img_name)

    # pre img
    pre_img = model.predict(test_img)[0]
    pre_img = np.minimum(np.maximum(pre_img, 0), 1)

    print ("pre_img .shape:{}".format(pre_img.shape))
    img_name = os.path.basename(imgs_name[ind]).split(".png")[0] + "_pre.png"
    write_noiseimg(pre_img * 255.0 , output_dir, img_name)


