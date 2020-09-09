#!/usr/bin/env python
# encoding: utf-8
from cnn.model_5layers import *
# from  model2 import  *
from data_prepare.tools import *
from data_prepare.data_generate import *
import numpy as np
import tensorflow as tf
import os

local_time = get_time()
# weights_path = "../dataset/checkpoint/"
# weights_path = '../checkpoint_list/UNet_5layers_200613_56117/UNet_5layers_200613_56117'
weights_path = "../checkpoint_list/checkpoint_UNet_5layers_200612_3092/UNet_5layers_200612_30902"
# input_dir = "./dataset/full_HD/original_mot16"
input_dir = "../dataset/caltechPedestrians/test"
weights_name = weights_path.split("/")[-1]
output_dir = os.path.join(input_dir, "output_{}".format(weights_name))
xls_path = os.path.join(output_dir, "psnr_{}.xlsx".format(local_time))
isexist(output_dir)

test_size = 512
lam_noise = 20
workbook, worksheet = create_workbook(xls_path)

test_imgs = load_data_images(input_dir)
noise_patch, gt_patch = random_crop(test_imgs, test_size, lam_noise)

input_size = (test_size,test_size, 3)
model, model_name = unet(pretrained_weight= weights_path,input_size=input_size)

data_psnr = {}
avg_psnr = 0.0
for idx, gt in enumerate(gt_patch):
    noise = noise_patch[idx]
    noise_test = np.expand_dims(noise, axis=0)
    print(noise_test.shape)
    pre_img = model(noise_test)[0]
    pre_img = np.minimum(np.maximum(pre_img, 0),1)

    noise_name = "{}_noise.jpg".format(idx)
    pre_name = "{}_pre.jpg".format(idx)
    write_img(noise*255,output_dir,noise_name)
    write_img(pre_img*255,output_dir,pre_name)

    # evaluate psnr
    im1 = tf.image.convert_image_dtype(gt, tf.float32)
    # im1 = tf.image.convert_image_dtype(ori_img, tf.float32)
    im2 = tf.image.convert_image_dtype(pre_img, tf.float32)
    curr_psnr = tf.image.psnr(im1, im2, max_val=1.0)
    avg_psnr += curr_psnr

    # write csv
    # name = (imgs_name[ind]).split(".png")[0]
    data_psnr.update({idx: curr_psnr})

avg_psnr = avg_psnr / len(gt_patch)
data_psnr.update({"avg_psnr": avg_psnr})

# write csv
row = 0
col = 0
for key, value in (data_psnr.items()):
    worksheet.write(row, col, key)
    worksheet.write(row, col + 1, value)
    row += 1

workbook.close()






