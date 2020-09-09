#!/usr/bin/env python
# encoding: utf-8
from data_prepare.tools import *
import tensorflow as tf
from data_prepare.tiles import *
from cnn.model_5layers import *
from data_prepare.change_img_size import *

data_dir_set =[]
imgs_dir = "../dataset/caltechPedestrians/test"
weights_path = "../checkpoint_list/checkpoint_UNet_5layers_200612_3092/UNet_5layers_200612_30902"

weights_name = os.path.basename(weights_path)
output_dir = os.path.join(imgs_dir, "output_{}_{}".format("min_tile",weights_name))
isexist(output_dir)

side_overlap_size = 320
overlap_size = 3
# tile_size_set = [320,640,1280]
tile_size_set = [256,160,128]

elect_px = 2
lam_noise = 20

# psnr
def compute_psnr(gt, pre):
    im1 = tf.image.convert_image_dtype(gt, tf.float32)
    im2 = tf.image.convert_image_dtype(pre, tf.float32)
    curr_psnr = tf.image.psnr(im1, im2, max_val=1.0)
    return curr_psnr

# tile_set2img
def tile_set2img(pre_tile_set,overlap_size):
    stitch_img = stitch_random_tiles(pre_tile_set, overlap_size)
    pre_img = np.minimum(np.maximum(stitch_img, 0), 1)
    return pre_img

# img2tile_set
def img2tile_set(noise_img, overlap_size, tile_size):
    overlap_noise_img = overlap_tile(overlap_size, noise_img)
    tile_img_set = split_images2_random_pathes(overlap_noise_img, overlap_size, tile_size)
    return tile_img_set

def predict_tile_set(model, noise_tile_set):
    pre_tile_img_set = []
    for h in range(len(noise_tile_set)):
        pre_tiles = []
        for w in range(len(noise_tile_set[0])):
            curr_tile = noise_tile_set[h][w]
            tile = np.expand_dims(curr_tile, axis=0)
            # write_img(curr_tile  * 255.0, output_dir, "tile_{}_{}{}.jpg".format( name,str(h),str(w)))

            pre_img = model.predict(tile)[0]
            # pretile = np.minimum(np.maximum(pre_img, 0), 1)
            # write_img(pretile* 255.0, output_dir, "pretile_{}_{}{}.jpg".format(name,str(h),str(w)))

            pre_tiles.append(pre_img)
        pre_tile_img_set.append(pre_tiles)
    return pre_tile_img_set

def get_img_HWC(img,overlap_size,tile_size):
    overlap_img = add_one_side43(img, side_overlap_size)
    tile_img_set = img2tile_set(overlap_img, overlap_size, tile_size)
    H = tile_img_set[0][0].shape[0]
    W = tile_img_set[0][0].shape[1]
    C = tile_img_set[0][0].shape[2]
    return H,W,C

if __name__ == '__main__':
    local_time = get_time()

    xls_path = os.path.join(output_dir, "psnr_{}.xlsx".format(local_time))
    workbook, worksheet = create_workbook(xls_path)

    ori_imgs, imgs_name = load_data_images(imgs_dir)
    noise_patch, gt_patch = add_noise_patch(ori_imgs, lam_noise)

    data_psnr = {}
    avg_psnr = 0.0
    for tile_size in tile_size_set:
        tile_size = tile_size

        img = gt_patch[0]
        H, W, C= get_img_HWC(img, overlap_size, tile_size)
        print("H:{} W:{} C:{}".format(H,W,C))
        model, model_name = unet(pretrained_weight=weights_path, input_size=(H,W,C))

        for ind, img in enumerate(imgs_name):
            name = imgs_name[ind].split(".")[0]
            gt_img = gt_patch[ind]

            noise_img = noise_patch[ind]
            overlap_noise_img = add_one_side43(noise_img, side_overlap_size)
            overlap_noise_img = np.array(overlap_noise_img).astype(np.float)

            noise_tile_img_set = img2tile_set(overlap_noise_img, overlap_size, tile_size)
            pre_tile_img_set = predict_tile_set(model, noise_tile_img_set)
            pre_img = tile_set2img(pre_tile_img_set,overlap_size)
            # add_img_name = "add_{}.jpg".format(name)
            # write_img(pre_img*255,output_dir,add_img_name)

            pre_img = remove_one_side43(pre_img,side_overlap_size)

            # pre_img_name = "pre_{}.jpg".format(name)
            # write_img(pre_img*255,output_dir,pre_img_name)
            curr_whole_psnr = compute_psnr(gt_img,pre_img)

            # elect test pixels
            ori_pixels = elect_pixels(gt_img , elect_px)
            stitch_pixels = elect_pixels(pre_img, elect_px)
            curr_part_psnr = compute_psnr(ori_pixels,stitch_pixels)

            data_psnr.update({str(H) + "_" + str(W) + "_" + str(overlap_size) + "_whole_" + name: curr_whole_psnr})
            data_psnr.update({str(H) + "_" + str(W) + "_" + str(overlap_size) + "_part_" + name: curr_part_psnr})

    # write csv
    row = 0
    col = 0
    for key, value in (data_psnr.items()):
        worksheet.write(row, col, key)
        worksheet.write(row, col + 1, value)
        row += 1

    workbook.close()

