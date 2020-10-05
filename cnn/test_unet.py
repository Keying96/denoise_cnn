#!/usr/bin/env python
# encoding: utf-8
from cnn.model_5layers import *
# from  model2 import  *
from data_prepare.tools import *
from data_prepare.data_generate import *
from data_prepare.create_noise_data import *

import numpy as np
import time

weights_path = "../checkpoint_list/checkpoint_UNet_5layers_200612_3092/UNet_5layers_200612_30902"
# weights_path = "/home/zhu/PycharmProjects/denoise_cnn/openvion/UNet_5layers_20-09-11-15-55"

# input_dir = "../dataset/caltechPedestrians/test"
input_dir = "../dataset/caltechPedestrians/parallel_test"
output_dir = "../dataset/caltechPedestrians/parallel_test/output_tile_single"
xls_path = os.path.join(output_dir, "single_runtime.xls")

isexist(output_dir)


# def run(image_names, test_imgs):
def run(input_dir, lam_noise, data_string):
    gt_imgs, image_names = load_data_images(input_dir)
    test_imgs = poisson_noise_imgs(gt_imgs,lam_noise)
    H, W, C = get_img_HWC(test_imgs)

    data_string.append(len(test_imgs)+1)

    model, model_name = unet(pretrained_weight=weights_path, input_size=(H, W, C))

    for ind, test_img in enumerate(test_imgs):
        test_img = np.expand_dims(test_img, axis=0)

        pre_img = model(test_img)[0]
        pre_img = np.minimum(np.maximum(pre_img, 0), 255)
        pre_name = image_names[ind].split(".") [0] + "_pre0924.jpg"
        print(pre_img.shape)
        print(pre_img)
        write_img(pre_img, output_dir, pre_name)



if __name__ == '__main__':
    lam_noise = 20

    # ImgNum, SleepTime, TileSize, Runtime
    data_string = []

    start_time = time.time()
    run(input_dir, lam_noise, data_string)
    run_time = time.time() -start_time
    data_string.append(run_time)
    print("the run time of single program is {}s".format(run_time))
    insert_worksheet(xls_path, data_string)




