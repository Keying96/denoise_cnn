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

save_freq = 500
"""
from model import *
from data_prepare.data_augment import *
from data_prepare.data_augment import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  LearningRateScheduler, ModelCheckpoint

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


gt_dir = "./dataset/myData/gt_test"
checkpoint_path = "./dataset/checkpoint/"

batch_size = 1
epochs = 15
SAVE_EVERY = 2
crop_size = 512
lam_max = 20

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img, cmap = "gray")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# custom losses
def l1_loss_function(y_true, y_pre):
    return tf.reduce_mean(tf.abs(y_true - y_pre))

def scheduler(epoch):
  if epoch < 2000:
    return 1e-4
  else:
    return 1e-5

def load_data():
    pass

if __name__ == '__main__':
    # data preparation
    train_images = load_images(gt_dir)
    print(train_images.shape)

    X = train_images[:-5]
    X_val = train_images[-5:]
    num_tr = len(X)

    print('%d training images'%len(X))
    print('%d validation images'%len(X_val))

    noise_patch, gt_patch = random_crop(X, crop_size, lam_max)
    train_gen = augment_images_generator(noise_patch, gt_patch, batch_size)

    noise_patch, gt_patch = random_crop(X_val, crop_size, lam_max)
    val_gen = augment_images_generator(noise_patch, gt_patch, batch_size)


    # create the model
    model = unet()

    # compile the model
    model.compile(optimizer = Adam(lr = 1e-4),
                  loss = l1_loss_function,
                  metrics = ["accuracy"])


    # model summary
    # model.summary()

    # train with data generator
    lr_scheduler  = LearningRateScheduler(scheduler)
    checkpointer = ModelCheckpoint(os.path.join(checkpoint_path,'model_{epoch:04d}.h5'), verbose=1,
                                   save_freq= SAVE_EVERY, save_best_only=True)

    history = model.fit_generator(generator= train_gen,
                                  steps_per_epoch = num_tr // batch_size,
                                  epochs= epochs,
                                  validation_data=val_gen,
                                  validation_steps= 1,
                                  callbacks= [lr_scheduler, checkpointer])

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
