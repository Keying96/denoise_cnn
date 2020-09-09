#!/usr/bin/env python
# encoding: utf-8
from data_prepare.tools import *
from data_prepare.create_noise_data import *
# from data_prepare.utils import *
import os
import numpy as np

# input_dir = "..\\dataset\\full_HD\\fullHD"
# output_dir = "..\\dataset\\full_HD\\fullHD\\out"
input_dir = "../dataset/super_resolution/sp_test/test"
output_dir = os.path.join(input_dir, "output_tiles/random_tiles")

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def elect_testimg_part(np_img, tile_size, ):
    return

def elect_pixels(np_img, elect_px):
    height = np_img.shape[0]
    weight = np_img.shape[1]
    dx = int(height/2)
    dy = int(weight/2)

    elect1 = np_img[:, dy-elect_px:dy+elect_px, :]
    elect1 = elect1.transpose((1,0,2))
    elect2 = np_img[dx-elect_px:dx+elect_px, 0:dy-elect_px, :]
    elect3 = np_img[dx-elect_px:dx+elect_px, dy+elect_px:dy, :]
    elect_pixels = np.hstack([elect1, elect2])
    elect_pixels = np.hstack([elect_pixels, elect3])
    return elect_pixels

def stitch_random_tiles(tiles,overlap_px):
    height = tiles[0][0].shape[0]
    weight = tiles[0][0].shape[1]

    stitch_tiles = []
    for h in range(len(tiles)):
        stitch_tile = []
        for w in range(len(tiles[0])):
            stitch_tile.append(tiles[h][w][overlap_px:height - overlap_px, overlap_px:weight - overlap_px, :])
        stitch_tiles.append(stitch_tile)

    vstack_tiles = []
    for h in range(len(tiles)):
        curr_vstack = stitch_tiles[h][0]
        for w in range(len(tiles[0])-1):
            curr_vstack = np.hstack([curr_vstack, stitch_tiles[h][w+1]])
        vstack_tiles.append(curr_vstack)

    stitch_img = vstack_tiles[0]
    for i in range(len(vstack_tiles)-1):
        # write_img(vstack_tiles[i], output_dir, "vstack_"+str(i)+ ".jpg")
        stitch_img = np.vstack([stitch_img, vstack_tiles[i+1]])

    return stitch_img

def stitch_16tiles(tiles,overlap_px):
    height = tiles[0].shape[0]
    weight = tiles[0].shape[1]

    stitch_tiles = []
    for ind in range(len(tiles)):
        stitch_tiles.append(tiles[ind][overlap_px:height-overlap_px, overlap_px:weight-overlap_px, :])

    vstack_tiles = []
    for ind in range(0,len(stitch_tiles),4):
        img1 = np.vstack([stitch_tiles[ind], stitch_tiles[ind+1]])
        img2 = np.vstack([stitch_tiles[ind+2], stitch_tiles[ind+3]])
        img = np.vstack([img1, img2])
        vstack_tiles.append(img)

    # for ind in range(len(hstack_tiles)):
    #     write_img(hstack_tiles[ind], output_dir, "hstack"+str(ind)+".jpg")

    img1 = np.hstack([vstack_tiles[0], vstack_tiles[1]])
    img2 = np.hstack([vstack_tiles[2], vstack_tiles[3]])
    stitch_img = np.hstack([img1,img2])

    return stitch_img

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

def split_images2_random_pathes(overlap_img, overlap_px, tile_number):
    height = overlap_img.shape[0]
    weight = overlap_img.shape[1]
    dx = int((height-2*overlap_px)/tile_number)
    dy = int((weight-2*overlap_px)/tile_number)
    # print("dx:{} dy:{}".format(dx, dy))

    tiles_set = []
    for h in range(tile_number):
        curr_tile_imgs = []
        for w in range(tile_number):
            curr_tile_img = overlap_img[h*dx:(h+1)*dx+2*overlap_px, w*dy:(w+1)*dy+2*overlap_px,:]
            curr_tile_imgs.append(curr_tile_img)
        tiles_set.append(curr_tile_imgs)
    return tiles_set

def split_images2_16pathes(overlap_img, overlap_px):
    height = overlap_img.shape[0]
    weight = overlap_img.shape[1]
    # dx = int((height)/4)
    # dy = int((weight)/4)
    dx = int((height-2*overlap_px)/4)
    dy = int((weight-2*overlap_px)/4)
    print("dx:{} dy:{}".format(dx, dy))

    tile1_1 = overlap_img[0:dx+2*overlap_px, 0:dy+ 2*overlap_px, :]
    tile1_2 = overlap_img[dx:2*dx+2*overlap_px, 0:dy+2*overlap_px, :]
    tile1_3 = overlap_img[2*dx:3*dx+2*overlap_px, 0:dy+2*overlap_px, :]
    tile1_4 = overlap_img[3*dx:height, 0:dy+2*overlap_px, :]

    tile2_1 = overlap_img[0:dx+2*overlap_px, dy:2*dy+2*overlap_px, :]
    tile2_2 = overlap_img[dx:2*dx+2*overlap_px, dy:2*dy+2*overlap_px, :]
    tile2_3 = overlap_img[2*dx:3*dx+2*overlap_px, dy:2*dy+2*overlap_px, :]
    tile2_4 = overlap_img[3*dx:height, dy:2*dy+2*overlap_px, :]

    tile3_1 = overlap_img[0:dx+2*overlap_px, 2*dy:3*dy+2*overlap_px, :]
    tile3_2 = overlap_img[dx:2*dx+2*overlap_px, 2*dy:3*dy+2*overlap_px, :]
    tile3_3 = overlap_img[2*dx:3*dx+2*overlap_px, 2*dy:3*dy+2*overlap_px, :]
    tile3_4 = overlap_img[3*dx:height, 2*dy:3*dy+2*overlap_px, :]

    tile4_1 = overlap_img[0:dx+2*overlap_px, 3*dy:weight, :]
    tile4_2 = overlap_img[dx:2*dx+2*overlap_px, 3*dy:weight, :]
    tile4_3 = overlap_img[2*dx:3*dx+2*overlap_px, 3*dy:weight, :]
    tile4_4 = overlap_img[3*dx:height, 3*dy:weight, :]

    # write_img(tile1, output_dir, "tile1.jpg")
    # write_img(tile2, output_dir, "tile2.jpg")
    # write_img(tile3, output_dir, "tile3.jpg")
    # write_img(tile4, output_dir, "tile4.jpg")

    tiles_set = [tile1_1]
    tiles_set.append(tile1_2)
    tiles_set.append(tile1_3)
    tiles_set.append(tile1_4)

    tiles_set.append(tile2_1)
    tiles_set.append(tile2_2)
    tiles_set.append(tile2_3)
    tiles_set.append(tile2_4)

    tiles_set.append(tile3_1)
    tiles_set.append(tile3_2)
    tiles_set.append(tile3_3)
    tiles_set.append(tile3_4)

    tiles_set.append(tile4_1)
    tiles_set.append(tile4_2)
    tiles_set.append(tile4_3)
    tiles_set.append(tile4_4)

    for ind in range(len(tiles_set)):
        print ("height:{} weight:{}".format(tiles_set[ind].shape[0], tiles_set[ind].shape[1]))
        write_img(tiles_set[ind], output_dir, str(ind)+".jpg")
    return tiles_set

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


def overlap_tile(overlap_px,ori_img):
    # print("overlap_px:{}".format(overlap_px))
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
    # plot_img(overlap_img)
    return overlap_img

def add_noise_patch(images_patch, lam_noise):
    size = len(images_patch)
    print (images_patch[0].shape)

    H = images_patch[0].shape[0]
    W = images_patch[0].shape[1]

    gt_patch = np.zeros((size, H, W, 3), dtype=images_patch.dtype)
    noise_patch = np.zeros((size, H, W, 3 ), dtype=images_patch.dtype)

    for ind in range(size):
        curr_gt = images_patch[ind]
        gt_patch[ind] = curr_gt / 255.0
        noise_patch[ind] = add_train_poisson_noise(curr_gt, lam_noise) / 255.0

    return  noise_patch, gt_patch

def load_data_images(data_dir):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    imgs_list = glob.glob(data_dir + "/*.png") # get name list of all .png files
    imgs_list.extend(glob.glob(data_dir + "/*.jpg"))
    imgs_list.extend(glob.glob(data_dir + "/*.JPG"))

    print("this is imgs list:{}".format(imgs_list))
    print ("the size of imgs: {}".format(len(imgs_list)))

    # initrialize
    images = []
    imgs_name = []

    for i in range(len(imgs_list)):
        img_path = imgs_list[i]
        images.append(imread(img_path)) # 0 is grayscale mode

        name = os.path.basename(imgs_list[i])
        imgs_name.append(name)

    print(len(images))
    images = np.array(images).astype(np.float32)
    # return  np.stack(images,axis=0)[:,:,:,None]
    return  np.stack(images,axis=0), imgs_name

if __name__ == '__main__':
    ori_imgs, name_imgs = load_images(input_dir)
    print(name_imgs)
    # print(ori_imgs[0].shape)

    # overlap
    overlap_px = 5
    tile_number = 120
    overlap = overlap_tile(overlap_px, ori_imgs[0], name_imgs[0])
    write_img(overlap, output_dir, "overlap.jpg")

    # tiles images
    # tiles_set = split_images2_4pathes(overlap, overlap_px)
    # tiles_set = split_images2_16pathes(overlap, overlap_px)
    tiles_set = split_images2_random_pathes(overlap, overlap_px, tile_number)
    print("tile image height:{} weight:{}".format(tiles_set[0][0].shape[0], tiles_set[0][0].shape[1] ))

    # stitch imgae
    # stitch_img = stitch_4tiles(tiles_set, overlap_px)
    # stitch_img = stitch_16tiles(tiles_set, overlap_px)
    # write_img(stitch_img, output_dir, "stitch_16.jpg")
    stitch_img = stitch_random_tiles(tiles_set, overlap_px)
    write_img(stitch_img, output_dir, "stitch_random.jpg")
    print ("final img height:{} weight:{}".format(stitch_img.shape[0], stitch_img.shape[1]))

    # elect_px = 200
    # ori_pixels = elect_pixels(stitch_img, elect_px)
    # print("ori_pixels shape:{}".format(ori_pixels.shape))
    # write_img(ori_pixels, output_dir, "ori_pixels.jpg")
