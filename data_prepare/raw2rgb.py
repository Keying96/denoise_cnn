#!/usr/bin/env python
# encoding: utf-8
import  os
import glob
import rawpy
import numpy as np
import imageio


gt_dir = "../dataset/Sony/long"
rgb_dir = "../dataset/Sony/rgb"

gt_files = glob.glob(gt_dir + '/0*.ARW')
gt_names = [os.path.basename(gt_file).split(".ARW")[0] for gt_file in gt_files]
gt_ids  = [int(os.path.basename(gt_file)[0:5]) for gt_file in gt_files]

def raw2rgb(gt_ids):
    for gt_path in gt_files:
        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        rgb = (np.float32(im / 65535.0)) * 255
        print(rgb.shape)
        img_name = os.path.basename(gt_path).split(".ARW")[0] + ".png"
        save_name = os.path.join(rgb_dir, img_name)
        print (save_name)
        imageio.imwrite(save_name, rgb.astype("uint8"))

raw2rgb(gt_ids)