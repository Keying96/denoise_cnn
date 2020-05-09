#!/usr/bin/env python
# encoding: utf-8

from cnn.model import *
from data_prepare.tiles import *
from data_prepare.create_noise_data import *
from data_prepare import tools
from estimate.calculate_psnr import *
import os

input_dir = "../dataset/super_resolution/sp_test/test"
output_dir = os.path.join(input_dir, "output_tiles")
weights_path = "../dataset/checkpoint/"
xls_path = os.path.join(output_dir, "psrn_tiles_{}.xlsx".format("model_9layers"))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

overlap_px = 40
lam_noise = 20
data_psnr = {}
avg_psnr = 0.0

workbook, worksheet = tools.create_workbook(xls_path)
ori_imgs, imgs_name = tools.load_images(input_dir)

model = unet()
model.load_weights(weights_path)

for ind, img in enumerate(ori_imgs):
    ori_img = img
    noise_img = add_train_poisson_noise(ori_img, lam_noise) / 255.0
    noise_img = np.array(noise_img).astype(np.float)

    # overlap image
    overlap_noise_img = overlap_tile(overlap_px,noise_img,imgs_name[ind])
    write_img(overlap_noise_img * 255.0, output_dir, "overlap.jpg")
    tiles_set = split_images2_4pathes(overlap_noise_img, overlap_px)

    H = tiles_set[0].shape[0]
    W = tiles_set[0].shape[1]
    C = tiles_set[0].shape[2]

    model = unet(input_size=(H, W, C))
    # model.load_weights(weights_path)
    model.summary()

    pre_tile_set = {}
    for ind, tile in enumerate(tiles_set):
        tile = np.expand_dims(tiles_set[ind], axis=0)
        pre_img = model.predict(tile)[0]
        pre_tile_set[ind] = pre_img

    stitch_img = stitch_4tiles(pre_tile_set, overlap_px)
    write_img(stitch_img * 255.0, output_dir, "stitch.jpg")

    # elect test pixels
    ori_pixels = elect_pixels(ori_img/255.0)
    stitch_pixels = elect_pixels(stitch_img)
    #
    curr_psrn = estimate_tf_psnr(ori_pixels, stitch_pixels)
    name = (imgs_name[ind]).split(".png")[0]
    data_psnr.update({name:curr_psrn})

avg_psnr = avg_psnr/ (len(avg_psnr))
data_psnr.update({"avg_psrn" : avg_psnr})

# write csv
row = 0
col = 0
for key, value in (data_psnr.items()):
    worksheet.write(row, col, key)
    worksheet.write(row, col + 1, value)
    row += 1

workbook.close()






