#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import  os
import imageio
import  glob
from PIL import Image

rgb_dir = "../dataset/Sony/rgb"
output_gt_dir = "../dataset/myData/gt"
output_train_dir = "../dataset/myData/train"
output_val_dir = "../dataset/myData/val"

def load_img(add_fn):

    try:
        input_path = add_fn
        image = Image.open(input_path)
        input_arr = np.asarray(image).astype(np.float32)
        H = input_arr.shape[0]
        W = input_arr.shape[1]
        print ("H: {}, W: {}".format(H,W))
    except:
        print("the invalid path is {}".format(input_path))
        os.remove(input_path)
        return None
    else:
        print ("load the path {}".format(input_path))
        return input_arr

def write_noiseimg(poisson_img, output_dir, input_name):
    output_dir = os.path.join(output_dir, input_name)
    imageio.imwrite(output_dir, poisson_img.astype("uint8"))

def create_gray_img(input_arr, output_gt_dir, gray_name):
    r =  input_arr[:, :, 0]
    g = input_arr[:, :, 1]
    b = input_arr[:, :, 2]

    add = pow(r,2.0) + pow(g, 2.0) + pow(b,2.0)
    division = add / 3.0
    w_poi = np.sqrt(division)
    ratio_poi = 4
    noise_poi = ratio_poi * np.sqrt(w_poi)

    write_noiseimg(noise_poi, output_gt_dir, gray_name)
    return noise_poi

# def augment_train_poisson(gray_img, output_train_dir,poisson_name):
#     lam_max = 1
#     chi_rng = np.random.uniform(low=0.001, high=lam_max, size = 1)
#     poisson_img = np.random.poisson(chi_rng*(gray_img+0.5))/chi_rng - 0.5
#     print ("chi_rng:{}".format(chi_rng))
#     # noisy_mask = np.random.poisson(gray_img)
#     # poisson_img = gray_img + noisy_mask
#     write_noiseimg(poisson_img, output_train_dir, poisson_name)
#
# def augment_val_poisson(gray_img, output_val_dir,poisson_name):
#     chi = 30.0
#     poisson_img = np.random.poisson(chi * (gray_img + 0.5)) / chi - 0.5
#     write_noiseimg(poisson_img, output_val_dir, poisson_name)


if __name__ == '__main__':
    # create "groundTruth" , "train", "val"
    image_fns = glob.glob(rgb_dir + "/*.png")
    # image_fns = ["../dataset/Sony/rgb/00001_00_10s.png"]
    for image_path in image_fns:
        input_name = os.path.basename(image_path).split(".png")[0]
        image_arr = load_img(image_path)

        if image_arr is not None:
            # 保存gray_img作为gt
            gray_name = input_name + "_gt"+ ".png"
            gray_img = create_gray_img(image_arr, output_gt_dir, gray_name)

            # #保存poisson_img作为input
            # poisson_train_name = input_name + "_train" + ".png"
            # poisson_train_img = augment_train_poisson(gray_img, output_train_dir, poisson_train_name)
            # poisson_val_name = input_name + "_val" + ".png"
            # poisson_val_img = augment_train_poisson(gray_img, output_val_dir, poisson_val_name)


