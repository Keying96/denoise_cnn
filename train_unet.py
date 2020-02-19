#!/usr/bin/env python
# encoding: utf-8
"""
# optimizer && loss function
G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
G_opt = tf.train.AdamOptimnizer(learning_rate=lr).minimize(G_loss)

#Learning rate
learning_rate = 1e-4
for epoch in range(lastepoch, 4001):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        learning_rate = 1e-5

ps = 512  # patch size for training
save_freq = 500

# crop
H = input_images[str(ratio)[0:3]][ind].shape[1]
W = input_images[str(ratio)[0:3]][ind].shape[2]

xx = np.random.randint(0, W - ps)
yy = np.random.randint(0, H - ps)
input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

"""
from model import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


gt_dir = "./dataset/train_groundTruth/"
train_dir = "./dataset/train/"
save_train_dir = "./dataset/save_train/train"
save_gt_dir = "./dataset/save_train/1"
checkpoint_path = "./dataset/checkpoint/"

num_gt = len(os.listdir(gt_dir))
num_tr = len(os.listdir(train_dir))
print ("total groundTruth images: {}".format(num_gt))
print ("total training images: {}". format(num_tr))
print ("--")

batch_size = 2
epochs = 2
IMG_HEIGHT = 512
IMG_WIDTH = 512
# rescale = 1./255
rescale = 1.
epochs_list = []
lr_list = []

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img, cmap = "gray")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def data_generator(train_dir, gt_dir, rescale, batch_size,
                   save_train_dir, IMG_HEIGHT, IMG_WIDTH):

    data_gen_args = dict(rescale = rescale,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   rotation_range = 270,)

    image_gen = ImageDataGenerator(**data_gen_args)
    gt_gen = ImageDataGenerator(**data_gen_args)

    train_data_gen = image_gen.flow_from_directory(batch_size= batch_size,
                                                   directory= train_dir,
                                                   classes=None,
                                                   class_mode= "input",
                                                   shuffle= False,
                                                   target_size= [IMG_HEIGHT, IMG_WIDTH],
                                                   color_mode= "grayscale",
                                                   save_to_dir= save_train_dir)

    gt_data_gen = gt_gen.flow_from_directory(batch_size= batch_size,
                                            directory= gt_dir,
                                            classes=None,
                                            class_mode= "input",
                                            shuffle= False,
                                            target_size=[IMG_HEIGHT, IMG_WIDTH],
                                            color_mode= "grayscale",
                                            save_to_dir=save_gt_dir)

    train_generator = zip(train_data_gen, gt_data_gen)
    for (train_img, pre_img) in train_generator:
        yield (train_img, pre_img)
    # train_generator = (pair for pair in zip(train_data_gen, gt_data_gen))
    # train_generator =  zip(train_data_gen, gt_data_gen)
    # yield train_generator
    # yield train_data_gen, gt_data_gen
    # yield  train_data_gen

def compute_ramped_down_lrate():
    # return  learning_rate
    pass

def l1_loss_function(y_true, y_pre):
    return tf.reduce_mean(tf.abs(y_true - y_pre))

if __name__ == '__main__':
    # data preparation
    train_generator = data_generator(train_dir, gt_dir, rescale, batch_size,
                                     save_train_dir, IMG_HEIGHT, IMG_WIDTH)
    # sample = next(train_generator)


    # create the model
    model = unet()
    # compile the model
    model.compile(
        # optimizer = "adam",
        optimizer = tf.keras.optimizers.Adam(lr = 1e-4),
                  loss = l1_loss_function,
                  # loss = tf.keras.losses.poisson,
                  metrics = ["accuracy"])
    # model summary
    model.summary()

    # train with data generator
    history = model.fit_generator(train_generator,
                        steps_per_epoch = num_tr // batch_size,
                        epochs= epochs)

    model.save_weights(checkpoint_path)

    # visualize training results
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
