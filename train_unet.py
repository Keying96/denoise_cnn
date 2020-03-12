#!/usr/bin/env python
# encoding: utf-8

from model import *
# from model2 import *
# from data_prepare.data_augment import *
from data_prepare.data_generate import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  LearningRateScheduler, ModelCheckpoint

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


# gt_dir = "./dataset/myData/gt"
checkpoint_path = "./dataset/checkpoint/"
rgb_dir = './dataset/Sony/rgb/'

batch_size = 3
epochs = 10
SAVE_EVERY = 2
crop_size = 512
lam_noise = 20


# custom losses
def l1_loss_function(y_true, y_pre):
    return tf.reduce_mean(tf.abs(y_true - y_pre))

def scheduler(epoch):
  if epoch > 2000:
    return 1e-5
  else:
      return 1e-4

def load_data(rgb_dir, crop_size, lam_noise, batch_size):

    train_images = load_images(rgb_dir)
    print(train_images.shape)

    X = train_images[:-5]
    X_val = train_images[-5:]

    num_tr = len(X)
    num_val = len(X_val)
    print('%d training images'%len(X))
    print('%d validation images'%len(X_val))

    # noise_patch, gt_patch = random_crop(images_patch, crop_size) / rescale
    noise_patch, gt_patch = random_crop(X, crop_size, lam_noise)
    train_gen = augment_images_generator(noise_patch, gt_patch, batch_size)

    noise_patch, gt_patch = random_crop(X_val, crop_size, lam_noise)
    val_gen = augment_images_generator(noise_patch, gt_patch, batch_size)

    return  train_gen, val_gen, num_tr, num_val

if __name__ == '__main__':
    # data preparation
    train_gen, val_gen, num_tr, num_val = load_data(rgb_dir, crop_size, lam_noise, batch_size )

    # create the model
    # W = train_gen[0].shape[0]
    # H = train_gen[0].shape[1]
    # C = train_gen[0].shape[2]
    # model = unet(input_size=(W, H, C))
    model = unet()

    # compile the model
    model.compile(optimizer = Adam(lr = 1e-4),
                  loss = l1_loss_function)
                  # metrics = ["accuracy"])


    # model summary
    model.summary()

    # train with data generator
    lr_scheduler  = LearningRateScheduler(scheduler)
    # checkpointer = ModelCheckpoint(os.path.join(checkpoint_path,'model_{epoch:04d}.h5'),
    #                                monitor='loss', save_best_only=True, overwrite=True)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpointer = ModelCheckpoint(checkpoint_path)


    history = model.fit_generator(generator= train_gen,
                                  steps_per_epoch = num_tr // batch_size,
                                  epochs= epochs,
                                  validation_data=val_gen,
                                  validation_steps= 2,
                                  callbacks= [lr_scheduler, checkpointer])

    model.save_weights(checkpoint_path)

    loss = history.history["loss"]
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)


    plt.subplot(1, 1, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
