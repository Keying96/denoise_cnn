#!/usr/bin/env python
# encoding: utf-8

from data_prepare.change_img_size import *
from cnn.model_5layers import *
from data_prepare.tiles import *
from data_prepare.create_noise_data import *
from estimate.calculate_psnr import *
from data_prepare import tools
import os

side_overlap_px = 240
overlap_px = 3
# tile_number = 2
tile_number_set = [540,2]
elect_px = 2
lam_noise = 20
data_psnr = {}

today, ti_day = tools.get_time()
input_dir = "../dataset/full_HD/fullHD"
# output_dir = os.path.join(input_dir,"minmal_tiles_{}".format(tile_number))
output_dir = os.path.join(input_dir,"minmal_tiles_{}".format(ti_day))


weights_path = "../dataset/checkpoint/"
xls_path = os.path.join(output_dir, "psnr_tiles_{}.xlsx".format(ti_day))

ori_imgs, imgs_name = tools.load_images(input_dir)
tools.isexist(output_dir)
workbook, worksheet = tools.create_workbook(xls_path)

for tile_number in tile_number_set:
    tile_number = tile_number
    print ("tile_number: {}".format(tile_number))
    for ind, img in enumerate(ori_imgs):
        #get name
        name = imgs_name[ind].split(".")[0]

        # create noise imgs
        ori_img = img
        noise_img = add_train_poisson_noise(ori_img, lam_noise) / 255.0
        noise_img = np.array(noise_img).astype(np.float)
        # write_img(noise_img * 255.0, output_dir, "noise_{}_{}.jpg".format(tile_number, name))

        # overlap one side
        overlap_noise_img = add_one_side(noise_img, side_overlap_px)
        # write_img(overlap_noise_img * 255.0, output_dir, "overlap_side_{}_{}.jpg".format(tile_number, name))

        # overlap image
        # overlap_noise_img = overlap_tile(overlap_px,overlap_noise_img,imgs_name[ind])
        overlap_noise_img = overlap_tile(overlap_px,overlap_noise_img)
        tiles_set = split_images2_random_pathes(overlap_noise_img, overlap_px, tile_number)
        # write_img(overlap_noise_img * 255.0, output_dir, "overlap_noise_{}_{}.jpg".format(tile_number, name))


        H = tiles_set[0][0].shape[0]
        W = tiles_set[0][0].shape[1]
        C = tiles_set[0][0].shape[2]

        model = unet(input_size=(H, W, C))
        model.load_weights(weights_path)

        pre_tile_set = []
        for h in range(len(tiles_set)):
            pre_tiles = []
            for w in range(len(tiles_set[0])):
                curr_tile = tiles_set[h][w]
                tile = np.expand_dims(curr_tile, axis=0)
                # write_img(curr_tile  * 255.0, output_dir, "tile_{}_{}_{}{}.jpg".format(tile_number, name,str(h),str(w)))

                pre_img = model.predict(tile)[0]
                pretile = np.minimum(np.maximum(pre_img, 0), 1)
                # write_img(pretile* 255.0, output_dir, "pretile_{}_{}_{}{}.jpg".format(tile_number, name,str(h),str(w)))

                pre_tiles.append(pre_img)
            pre_tile_set.append(pre_tiles)

        stitch_img = stitch_random_tiles(pre_tile_set, overlap_px)
        stitch_img = np.minimum(np.maximum(stitch_img, 0), 1)
        stitch_img = remove_one_side(stitch_img, side_overlap_px)
        write_img(stitch_img* 255.0, output_dir, "stitch_img{}_{}_{}{}.jpg".format(tile_number, name,str(h),str(w)))

        # elect test pixels
        ori_pixels = elect_pixels(ori_img/255.0, elect_px)
        stitch_pixels = elect_pixels(stitch_img, elect_px)

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

