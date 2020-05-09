#!/usr/bin/env python
# encoding: utf-8
from cnn.model import *
# from  model2 import  *
from imageio import imread
from data_prepare.create_noise_data import  *

import  numpy as  np
import  tensorflow as tf
import os
import imageio
import glob
import xlsxwriter
"""
test the average psnr of image data set
"""
# weights_path = "./dataset/checkpoint_20200303_2/"
weights_path = "./dataset/checkpoint/"
# test_dir = "./dataset/super_resolution/sp8"
# test_dir = "./dataset/full_HD/mot16"
test_dir = "./dataset/super_resolution/sp_test"

# output_dir = "./dataset/Sony/output_20200312_1/"
# output_dir = "./dataset/super_resolution/sp8/output"
# output_dir = "./dataset/full_HD/mot16/output"
output_dir = "./dataset/super_resolution/sp_test/output_psrn"

pnsr_path = os.path.join(output_dir, "pnsr_{}.xlsx".format("model_9layers"))

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

workbook = xlsxwriter.Workbook(pnsr_path)
worksheet = workbook.add_worksheet()

lam_noise = 20

def write_noiseimg(poisson_img, output_dir, input_name):
    output_dir = os.path.join(output_dir, input_name)
    imageio.imwrite(output_dir, poisson_img.astype("uint8"))

""" Load dataset"""
def load_images(data_dir):
    imgs_list = glob.glob(data_dir + "/*.png")  # get name list of all .png files
    ori_imgs = []
    imgs_name = []

    for i in range(len(imgs_list)):
        img_path = imgs_list[i]
        ori_img = imread(img_path)

        ori_imgs.append(ori_img)

        name = os.path.basename(imgs_list[i])
        imgs_name.append(name)

    return  ori_imgs, imgs_name

ori_imgs, imgs_name = load_images(test_dir)
print (len(ori_imgs))
print (imgs_name)

data_psnr = {}
avg_psnr = 0.0
for ind, img in enumerate(ori_imgs):
    # load ori_image and get noise imga
    ori_img = img
    noise_img = add_train_poisson_noise(ori_img, lam_noise) / 255.0
    noise_img = np.array(noise_img).astype(np.float32)
    test_img = np.expand_dims(noise_img,axis=0)
    print ("test image shape:{}".format(test_img.shape))
    print("ori:{}".format(ori_img))
    print("test:{}".format(test_img))
    # create cnn_model and load the weights
    H = test_img.shape[1]
    W = test_img.shape[2]
    C = test_img.shape[3]

    model = unet(input_size=(H, W, C))
    model.load_weights(weights_path)
    model.summary()

    # test image
    img_name = os.path.basename(imgs_name[ind]).split(".png")[0] + "_test.png"
    # write_noiseimg(test_img[0] * 255.0 , output_dir, img_name)

    # pre img
    pre_img = model.predict(test_img)[0]
    pre_img = np.minimum(np.maximum(pre_img, 0), 1)
    print ("pre_img .shape:{}".format(pre_img.shape))
    img_name = os.path.basename(imgs_name[ind]).split(".png")[0] + "_pre.png"
    # write_noiseimg(pre_img * 255.0 , output_dir, img_name)

    # evaluate psnr
    # im1 = tf.image.convert_image_dtype(test_img[0], tf.float32)
    im1 = tf.image.convert_image_dtype(ori_img, tf.float32)
    # im1 = tf.image.convert_image_dtype(ori_img, tf.float32)
    im2 = tf.image.convert_image_dtype(pre_img, tf.float32)
    curr_psnr = tf.image.psnr(im1, im2, max_val=1.0)
    avg_psnr += curr_psnr

    # write csv
    name = (imgs_name[ind]).split(".png")[0]
    data_psnr.update({name: curr_psnr})

avg_psnr = avg_psnr / len(ori_imgs)
data_psnr.update({"avg_psnr" : avg_psnr})

# write csv
row = 0
col = 0
for key, value in (data_psnr.items()):
    worksheet.write(row, col, key)
    worksheet.write(row, col + 1, value)
    row += 1

workbook.close()




