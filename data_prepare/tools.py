#!/usr/bin/env python
# encoding: utf-8
import glob
from imageio import imread
import os
import imageio
import xlsxwriter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_img(np_img):
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()

def create_workbook(book_path):
    workbook = xlsxwriter.Workbook(book_path)
    worksheet = workbook.add_worksheet()

    return workbook, worksheet

def write_img(img, output_dir, input_name):
    output_dir = os.path.join(output_dir, input_name)
    imageio.imwrite(output_dir, img.astype("uint8"))

def load_images(data_dir):
    imgs_list = glob.glob(data_dir + "/*.png")  # get name list of all .png files
    imgs_list += (glob.glob(data_dir + "/*.jpg"))

    ori_imgs = []
    imgs_name = []

    for i in range(len(imgs_list)):
        img_path = imgs_list[i]
        ori_img = imread(img_path)

        ori_imgs.append(ori_img)

        name = os.path.basename(imgs_list[i])
        imgs_name.append(name)

    return  ori_imgs, imgs_name