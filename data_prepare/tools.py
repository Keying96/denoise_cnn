#!/usr/bin/env python
# encoding: utf-8
import glob
from imageio import imread
import os
import imageio
import xlsxwriter
import xlrd
import xlwt
from xlutils.copy import copy

import matplotlib.pyplot as plt
import time

def get_img_HWC(gt_patch):
    test_img = gt_patch[0]
    H = test_img.shape[0]
    W = test_img.shape[1]
    C = test_img.shape[2]
    return H,W,C

def get_time():
    local_time = time.localtime()
    data_format_localtime = time.strftime("%y-%m-%d-%H-%M", local_time)
    print(data_format_localtime)
    return data_format_localtime

def isexist(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def crop_img(img, crop_height, crop_width):
    crop_img= img[0:crop_height, 0:crop_width,:]
    return crop_img

def plot_img(np_img):
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()

def create_workbook(book_path):
    workbook = xlsxwriter.Workbook(book_path)
    worksheet = workbook.add_worksheet()
    return workbook, worksheet

def insert_worksheet(xls_path, data):
    if not os.path.exists(xls_path):
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("multi")
        title_strings = ["ImgNum", "SleepTime", "TileSize", "RunTime"]
        len_string = len(title_strings)
        for i in range(len_string):
            sheet.write(0, i, title_strings[i])
        workbook.save(xls_path)

    index = len(data)
    workbook = xlrd.open_workbook(xls_path)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[0])

    rows_old = worksheet.nrows
    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(0)

    for i in range(0, index):
        new_worksheet.write(rows_old, i, data[i])

    new_workbook.save(xls_path)

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

if __name__ == '__main__':
    # get_time()
    output_dir = "../dataset/caltechPedestrians/parallel_test/output_tile"
    xls_path = os.path.join(output_dir, "multi_runtime.xls")
    print(xls_path)
    data = [100, 39, 69]
    insert_worksheet(xls_path, data)

