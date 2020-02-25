#!/usr/bin/env python
# encoding: utf-8
import os
import imageio
import  numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

gt_dir = "../dataset/tmp/gt/"
batch_size = 5
epochs = 15
crop_size = 512
rescale = 1.
save_train_dir = '../dataset/tmp/train'

def write_noiseimg(poisson_img, output_dir, input_name):
    output_dir = os.path.join(output_dir, input_name)
    imageio.imwrite(output_dir, poisson_img.astype("uint8"))

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        print (img.shape)
        img = np.squeeze(img)
        ax.imshow(img, "gray")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def data_generator(gt_dir, rescale, batch_size,
                   save_train_dir, crop_size):
    # train_data_gen, gt_data_gen = data_generator(train_dir, gt_dir, rescale, batch_size, save_train_dir, IMG_HEIGHT, IMG_WIDTH)
    # (x_train, y_train) = load_data()
    data_gen_args = dict(rescale = rescale,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   rotation_range = 270,)

    gt_gen = ImageDataGenerator(**data_gen_args)

    gt_data_gen = gt_gen.flow_from_directory(batch_size= batch_size,
                                                   directory= gt_dir,
                                                   classes=None,
                                                   class_mode= "input",
                                                   shuffle= False,
                                                   target_size= [crop_size, crop_size],
                                                   color_mode= "grayscale",
                                                   save_to_dir= save_train_dir)


    yield  gt_data_gen

    # train_data_gen = gt_data_gen
    #
    # for ind in range(batch_size):
    #     print (gt_data_gen.shape)
    #     gt_data_gen[ind] = gt_data_gen[ind, :, :, :]
    #     noisy_mask = np.random.poisson(gt_data_gen[ind])
    #     train_data_gen[ind] = gt_data_gen[ind] + noisy_mask
    #     write_noiseimg(train_data_gen[ind], save_train_dir, "noise_{}.png".format(ind))
    #
    # return train_data_gen, gt_data_gen

train_data_gen = data_generator(gt_dir, rescale, batch_size,save_train_dir, crop_size)
sample_training_images = next(train_data_gen)
plotImages(sample_training_images[1][0])
plotImages(sample_training_images[0][0])



#
# train_data_gen, _ = data_generator(gt_dir, rescale, batch_size,save_train_dir, crop_size)
# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)


