#!/usr/bin/env python
# encoding: utf-8
from data_prepare.tools import *
# from data_prepare.utils import *
import os
import numpy as np

# input_dir = "..\\dataset\\full_HD\\fullHD"
# output_dir = "..\\dataset\\full_HD\\fullHD\\out"
input_dir = "../dataset/super_resolution/sp_test/test"
output_dir = os.path.join(input_dir, "output_tiles")

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# def select_pixels(image):
#     return pixels

def stitch_4tiles(tiles,overlap_px):
    height = tiles[0].shape[0]
    weight = tiles[0].shape[1]

    stitch1 = tiles[0][overlap_px:height-overlap_px, overlap_px:weight-overlap_px, :] # top_left
    stitch2 = tiles[1][overlap_px:height-overlap_px, overlap_px:weight-overlap_px, :] # top_right
    stitch3 = tiles[2][overlap_px:height-overlap_px, overlap_px:weight-overlap_px, :] # bottom_left
    stitch4 = tiles[3][overlap_px:height-overlap_px, overlap_px:weight-overlap_px, :] # bottom_right

    img1 = np.hstack([stitch1, stitch2])
    img2 = np.hstack([stitch3, stitch4])
    stitch_img = np.vstack([img1,img2])

    return stitch_img

def split_images2_4pathes(overlap_img, overlap_px):
    height = overlap_img.shape[0]
    weight = overlap_img.shape[1]
    dx = int(height/2)
    dy = int(weight/2)

    tile1 = overlap_img[0:dx+overlap_px, 0:dy+overlap_px, :]
    tile2 = overlap_img[0:dx+overlap_px, dy-overlap_px:weight, :]
    tile3 = overlap_img[dx-overlap_px:height, 0:dy+overlap_px, :]
    tile4 = overlap_img[dx-overlap_px:height, dy-overlap_px:weight, :]

    # write_img(tile1, output_dir, "tile1.jpg")
    # write_img(tile2, output_dir, "tile2.jpg")
    # write_img(tile3, output_dir, "tile3.jpg")
    # write_img(tile4, output_dir, "tile4.jpg")

    tiles_set = [tile1]
    tiles_set.append(tile2)
    tiles_set.append(tile3)
    tiles_set.append(tile4)
    return tiles_set


def overlap_tile(overlap_px,ori_img, name_img):
    print("overlap_px:{}".format(overlap_px))
    # flip(m,0) equivalent to flipud(m)
    # flip(m,1) equivalent to fliplr(m)
    def _flip_vertically(np_array):
        return np.flip(np_array, 0)

    def _flip_horizontally(np_array):
        return np.flip(np_array, 1)

    ori_height = ori_img.shape[0]
    ori_weight = ori_img.shape[1]

    # expand height
    top = _flip_vertically(ori_img[0:overlap_px, 0:ori_weight,:])
    bottom = _flip_vertically(ori_img[ori_height-overlap_px: ori_height, 0:ori_weight, :])

    img1 = np.vstack([top, ori_img])
    img2 = np.vstack([img1, bottom])

    # expand width
    img2_height = img2.shape[0]
    img2_weight = img2.shape[1]
    left = _flip_horizontally(img2[0:img2_height, 0:overlap_px, :])
    right = _flip_horizontally(img2[0:img2_height, img2_weight-overlap_px:img2_weight, :])

    img3 = np.hstack([left, img2])
    overlap_img = np.hstack([img3, right])
    plot_img(overlap_img)
    return overlap_img


if __name__ == '__main__':
    ori_imgs, name_imgs = load_images(input_dir)
    print(name_imgs)
    print(ori_imgs[0].shape)

    overlap_px = 50
    overlap = overlap_tile(overlap_px, ori_imgs[0], name_imgs[0])
    write_img(overlap, output_dir, "overlap.jpg")
    tiles_set = split_images2_4pathes(overlap, overlap_px)

    stitch_img = stitch_4tiles(tiles_set, overlap_px)
    write_img(stitch_img, output_dir, "stitch.jpg")