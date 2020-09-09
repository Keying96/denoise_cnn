#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def add_one_side43(ori_img, overlap_size):
    def _flip_vertically(np_array):
        return np.flip(np_array, 0)

    ori_height = ori_img.shape[0]
    ori_weight = ori_img.shape[1]

    bottom = _flip_vertically(ori_img[ori_height-overlap_size:ori_height, :, :])
    overlap_img = np.vstack([ori_img, bottom])
    return overlap_img

def remove_one_side43(ori_img, overlap_size):

    ori_height = ori_img.shape[0]

    remove_img = ori_img[0:ori_height-overlap_size,:,:]
    return remove_img

def add_one_side(ori_img, overlap_size):
    def _flip_horizontally(np_array):
        return np.flip(np_array, 1)

    ori_height = ori_img.shape[0]
    ori_weight = ori_img.shape[1]

    right =_flip_horizontally(ori_img[0:ori_height, ori_weight-overlap_size:ori_weight,:])
    overlap_img = np.hstack([ori_img, right])
    return overlap_img

def remove_one_side(ori_img, overlap_size):

    ori_weight = ori_img.shape[1]

    remove_img = ori_img[:,0:ori_weight-overlap_size,:]
    return remove_img