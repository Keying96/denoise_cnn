#!/usr/bin/env python
# encoding: utf-8
import os
from data_prepare import tools
from data_prepare.create_noise_data import *
from estimate.calculate_psnr import *
from cnn.model_5layers import *
from data_prepare.tools import *

input_dir = "../dataset/super_resolution/sp_test/test"
output_dir = os.path.join(input_dir, "output_tiles")
weights_path = "../dataset/checkpoint/"
xls_path = os.path.join(output_dir, "psrn_tiles_{}.xlsx".format("model_9layers"))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

lam_noise = 20
crop_height = 1
crop_width = 1
# workbook, worksheet = tools.create_workbook(xls_path)
ori_imgs, imgs_name = tools.load_images(input_dir)

model = unet()
model.load_weights(weights_path)

# data_psnr = {}
for ind, img in enumerate(ori_imgs):
    # load ori_image and get noise imga
    # ori_img = img
    ori_img = tools.crop_img(img, crop_height, crop_width)
    print ("crop img width:{} height:{}".format(ori_img.shape[0], ori_img.shape[1]))
    noise_img = add_train_poisson_noise(ori_img, lam_noise) / 255.0
    noise_img = np.array(noise_img).astype(np.float32)
    test_img = np.expand_dims(noise_img,axis=0)
    print ("test image shape:{}".format(test_img.shape))

    # create cnn_model and load the weights
    H = test_img.shape[1]
    W = test_img.shape[2]
    C = test_img.shape[3]
    print ("H:{} W:{} C:{}".format(H, W, C))

    model = unet(input_size=(H, W, C))
    model.load_weights(weights_path)
    # model.summary()

    # test image
    # img_name = os.path.basename(imgs_name[ind]).split(".png")[0] + "_test.png"
    # # write_noiseimg(test_img[0] * 255.0 , output_dir, img_name)
    img_name = os.path.basename(imgs_name[ind]).split(".png")[0] + "_test_crop.png"
    write_img(test_img[0] * 255.0 , output_dir, img_name)

    # pre img
    pre_img = model.predict(test_img)[0]
    pre_img = np.minimum(np.maximum(pre_img, 0), 1)
    print ("pre_img .shape:{}".format(pre_img.shape))
    # img_name = os.path.basename(imgs_name[ind]).split(".png")[0] + "_pre.png"
    # write_noiseimg(pre_img * 255.0 , output_dir, img_name)
    img_name = os.path.basename(imgs_name[ind]).split(".png")[0] + "_pre_crop.png"
    write_img(pre_img * 255.0 , output_dir, img_name)


#     # evaluate psnr
#     im1 = noise_img
#     im2 = img / 255.0
#     im3 = pre_img
#     print("noise image im1:{}".format(im1[400,400,:]))
#     print("ori image im2:{}".format(im2[400,400,:]))
#     print("denoise image im3:{}".format(im3[400,400,:]))
#
#     # same = estimate_tf_psnr(im2, im2)
#     psnr = estimate_tf_psnr(im1, im2)
#     noise_psnr = estimate_tf_psnr(im1,im3)
#     clear_psnr = estimate_tf_psnr(im2,im3)
#     # print ("same psnr:{}".format(same))
#     print ("psnr:{} noise_psnr:{} clear_psnr:{}".format(psnr, noise_psnr, clear_psnr))
#
#     # avg_ssim += curr_ssim
#
#     # write csv
#     name = (imgs_name[ind]).split(".png")[0]
#     data_psnr.update({name + "_noise": noise_psnr})
#     data_psnr.update({name + "_clear": clear_psnr})


# avg_ssim = avg_ssim / len(ori_imgs)
# data_ssim.update({"avg_ssim" : avg_ssim})

# # write csv
# row = 0
# col = 0
# for key, value in (data_psnr.items()):
#     worksheet.write(row, col, key)
#     worksheet.write(row, col + 1, value)
#     row += 1
#
# workbook.close()
