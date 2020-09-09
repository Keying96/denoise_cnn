#!/usr/bin/env python
# encoding: utf-8

from cnn.model_5layers import *
# from model2 import *
# from data_prepare.data_augment import *
from data_prepare.data_generate import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  LearningRateScheduler, ModelCheckpoint
from data_prepare.tools import *

import tensorflow as tf
import os
import matplotlib.pyplot as plt

local_time = get_time()

# gt_dir = "./dataset/myData/gt"
checkpoint_dir = "../checkpoint_list/"

train_data_dir = '../dataset/Sony/train_data/'
# train_data_dir = "../dataset/caltechPedestrians/Original_train/"
# train_data_dir = '../dataset/Sony/test/'


batch_size = 5
epochs = 10
SAVE_EVERY = 2
crop_size = 512
lam_noise = 20
# val_number = 234
val_number = 50

# custom losses
def l1_loss_function(y_true, y_pre):
    return tf.reduce_mean(tf.abs(y_true - y_pre))

def scheduler(epoch):
  if epoch > 2000:
    return 1e-5
  else:
      return 1e-4

def load_data(train_data_dir, crop_size, lam_noise, batch_size):

    train_images = load_data_images(train_data_dir)
    # print(train_images)
    print(train_images.shape)

    X = train_images[:-val_number]
    X_val = train_images[-val_number:]

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
    train_gen, val_gen, num_tr, num_val = \
        load_data(train_data_dir, crop_size, lam_noise, batch_size )

    model, model_name = unet()

    # compile the cnn_model
    model.compile(optimizer = Adam(lr = 1e-4),
                  loss = l1_loss_function)


    # cnn_model summary
    model.summary()

    # train with data generator
    lr_scheduler  = LearningRateScheduler(scheduler)

    # checkpoint: network name, last loss train time
    checkpoint_dir = os.path.join(checkpoint_dir,"{}_{}".format(model_name,local_time))
    checkpoint_path = os.path.join(checkpoint_dir,"{}_{}".format(model_name,local_time))
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
    fig_path = os.path.join(checkpoint_path, "train_result.png")
    plt.savefig(fig_path)
    plt.show()
