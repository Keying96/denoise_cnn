#!/usr/bin/env python
# encoding: utf-8

from cnn.model_5layers import *
from data_prepare.tiles import *
from data_prepare.create_noise_data import *
from data_prepare import tools
from estimate.calculate_psnr import *
import os

# overlap_px_list = [5,4,3,2,1,0]
overlap_px = 5
# tile_number = 120
# tile_number_set =[30,24,20]
# tile_number_set =[1,2,3]
tile_number = 1
elect_px = 2
lam_noise = 20
data_psnr = {}

# input_dir = "../dataset/super_resolution/sp_test/test"
# input_dir = "../dataset/full_HD/original_mot16/crop120"
input_dir = "../dataset/full_HD/original_mot16/test"


output_dir = os.path.join(input_dir, "output_tiles")
weights_path = "../dataset/checkpoint/"
xls_path = os.path.join(output_dir, "psnr_tiles_{}.xlsx".format("overlaps_tiles0528"))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


workbook, worksheet = tools.create_workbook(xls_path)
ori_imgs, imgs_name = tools.load_images(input_dir)

# for tile_number in tile_number_set:
#     tile_number  = tile_number
for ind, img in enumerate(ori_imgs):
    img_name = imgs_name[ind].split(".jpg")[0]

    ori_img = img
    noise_img = add_train_poisson_noise(ori_img, lam_noise) / 255.0
    noise_img = np.array(noise_img).astype(np.float)

    # overlap image
    overlap_noise_img = overlap_tile(overlap_px,noise_img,imgs_name[ind])
    write_img(overlap_noise_img * 255.0, output_dir, "overlap_noise_{}_{}.jpg".format(tile_number, img_name))
    # tiles_set = split_images2_4pathes(overlap_noise_img, overlap_px)
    # tiles_set = split_images2_16pathes(overlap_noise_img, overlap_px)
    tiles_set = split_images2_random_pathes(overlap_noise_img, overlap_px, tile_number)


    H = tiles_set[0][0].shape[0]
    W = tiles_set[0][0].shape[1]
    C = tiles_set[0][0].shape[2]

    model = unet(input_size=(H, W, C))
    model.load_weights(weights_path)
    # model.summary()

    pre_tile_set = []
    # for id, tile in enumerate(tiles_set):
    #     tile = np.expand_dims(tiles_set[id], axis=0)
    #     pre_img = model.predict(tile)[0]
    #     pre_tile_set.append(pre_img)
    for h in range(len(tiles_set)):
        pre_tiles = []
        for w in range(len(tiles_set[0])):
            tile = np.expand_dims(tiles_set[h][w], axis=0)
            write_img(tiles_set[h][w] * 255.0, output_dir, "tile_{}_{}_{}{}.jpg".format(tile_number, img_name,str(h),str(w)))
            pre_img = model.predict(tile)[0]
            write_img(np.minimum(np.maximum(pre_img, 0), 1)* 255.0, output_dir, "pretile_{}_{}_{}{}.jpg".format(tile_number, img_name,str(h),str(w)))
            pre_tiles.append(pre_img)
        pre_tile_set.append(pre_tiles)

    # stitch_img = stitch_4tiles(pre_tile_set, overlap_px)
    # stitch_img = stitch_16tiles(pre_tile_set, overlap_px)
    stitch_img = stitch_random_tiles(pre_tile_set, overlap_px)
    stitch_img = np.minimum(np.maximum(stitch_img, 0), 1)
    write_img(stitch_img * 255.0, output_dir, "stitch_{}_{}.jpg".format(tile_number, img_name))

    # elect test pixels
    ori_pixels = elect_pixels(ori_img/255.0, elect_px)
    stitch_pixels = elect_pixels(stitch_img, elect_px)
    #
    name = (imgs_name[ind]).split(".png")[0]
    whole_psnr = estimate_tf_psnr(ori_img, stitch_img)
    part_psnr = estimate_tf_psnr(ori_pixels, stitch_pixels)
    data_psnr.update({str(H)+"_"+str(W)+"_"+str(overlap_px)+ "_whole_" + name:whole_psnr})
    data_psnr.update({str(H)+"_"+str(W)+"_"+str(overlap_px) + "_part_" + name:part_psnr})


# write csv
row = 0
col = 0
for key, value in (data_psnr.items()):
    worksheet.write(row, col, key)
    worksheet.write(row, col + 1, value)
    row += 1

workbook.close()






