#!/usr/bin/env python
# encoding: utf-8
from PIL import Image
from model import *

import os
import numpy as np

# img_dir = "./dataset/train/data/"
# img_name = "10006_SID_train.png"

# img_path = os.path.join(img_dir, img_name)
# image = Image.open(img_path)
# input_arr = np.asarray(image).astype(np.float32)
# print (input_arr)
#
# max = input_arr.max()
# min = input_arr.min()
# print ("max:{}, min:{}".format(max, min))
# normalized = (input_arr-(input_arr.min()))/((input_arr.max())-(input_arr.min()))
# print(normalized)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  LearningRateScheduler

batch_size = 2
epochs = 15
IMG_HEIGHT = 512
IMG_WIDTH = 512
# rescale = 1./255
rescale = 1.
epochs_list = []
lr_list = []

gt_dir = "./dataset/train_groundTruth/"
train_dir = "./dataset/train/"
save_train_dir = "./dataset/train_save/train"
save_gt_dir = "./dataset/train_save/1"
checkpoint_path = "./dataset/checkpoint/"


num_gt = len(os.listdir(gt_dir))
num_tr = len(os.listdir(train_dir))
print ("total groundTruth images: {}".format(num_gt))
print ("total training images: {}". format(num_tr))
print ("--")


def data_generator(train_dir, gt_dir, rescale, batch_size,
                   save_train_dir, IMG_HEIGHT, IMG_WIDTH):
    # train_data_gen, gt_data_gen = data_generator(train_dir, gt_dir, rescale, batch_size, save_train_dir, IMG_HEIGHT, IMG_WIDTH)
    # (x_train, y_train) = load_data()
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

    # train_generator = zip(train_data_gen, gt_data_gen)
    # for (train_img, pre_img) in train_generator:
    #     yield (train_img, pre_img)
    yield train_data_gen, gt_data_gen

# custom losses
def l1_loss_function(y_true, y_pre):
    return tf.reduce_mean(tf.abs(y_true - y_pre))

def scheduler(epoch):
  if epoch < 2000:
    return 1e-4
  else:
    return 1e-5


data_gen_args = dict(rescale=rescale,
                     horizontal_flip=True,
                     vertical_flip=True,
                     rotation_range=270, )

image_gen = ImageDataGenerator(**data_gen_args)
gt_gen = ImageDataGenerator(**data_gen_args)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               classes=None,
                                               class_mode="input",
                                               shuffle=False,
                                               target_size=[IMG_HEIGHT, IMG_WIDTH],
                                               color_mode="grayscale",
                                               save_to_dir=save_train_dir)

gt_data_gen = gt_gen.flow_from_directory(batch_size=batch_size,
                                         directory=gt_dir,
                                         classes=None,
                                         class_mode="input",
                                         shuffle=False,
                                         target_size=[IMG_HEIGHT, IMG_WIDTH],
                                         color_mode="grayscale",
                                         save_to_dir=save_gt_dir)

# sample_train_imgs = next(train_data_gen)
sample_gt_imgs = next(gt_data_gen)
train_generator = zip(train_data_gen, gt_data_gen)

# create the model
model = unet()

# compile the model
model.compile(optimizer=Adam(lr=1e-4),
              loss=l1_loss_function,
              metrics=["accuracy"])

# model summary
# model.summary()

# train with data generator
callback = LearningRateScheduler(scheduler)
history = model.fit_generator((train_data_gen, gt_data_gen),
                              # callbacks=callback,
                              steps_per_epoch=num_tr // batch_size,
                              epochs=epochs)

model.save_weights(checkpoint_path)
