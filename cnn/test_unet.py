#!/usr/bin/env python
# encoding: utf-8
# from cnn_model import *
from cnn.model import  *
from imageio import imread

import  numpy as  np
import os
import imageio
import glob

weights_path = "./dataset/checkpoint/"
test_dir = "./dataset/Sony/tmp_rgb"
output_dir = "./dataset/Sony/output"

def write_noiseimg(poisson_img, output_dir, input_name):
    output_dir = os.path.join(output_dir, input_name)
    imageio.imwrite(output_dir, poisson_img.astype("uint8"))


""" Load test images"""
def load_images(data_dir):
    imgs_list = glob.glob(data_dir + "/*.png")  # get name list of all .png files
    images = []
    imgs_name = []

    for i in range(len(imgs_list)):
        img_path = imgs_list[i]
        ori_img = imread(img_path)
        noisy_mask = np.random.poisson(ori_img)
        poi_img = noisy_mask + ori_img
        images.append(poi_img) # 0 is grayscale mode
        name = os.path.basename(imgs_list[i])
        imgs_name.append(name)

    images = np.array(images).astype(np.float32)
    return  np.stack(images,axis=0)[:,:,:,None], imgs_name
    # return  np.stack(images,axis=0), imgs_name

""" Denoise test images"""
test_set, imgs_name = load_images(test_dir)

""" Re-create the cnn_model and load the weights """
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
    img_name = os.path.basename(imgs_name[ind]).split("_gt")[0] + "_test.png"
    write_noiseimg(test_img[0], output_dir, img_name)
    pre_img = model.predict(test_img)[0]
    print ("pre_img .shape:{}".format(pre_img.shape))
    img_name = os.path.basename(imgs_name[ind]).split("_gt")[0] + "_pre.png"
    write_noiseimg(pre_img, output_dir, img_name)


