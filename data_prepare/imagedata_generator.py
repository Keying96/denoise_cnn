#!/usr/bin/env python
# encoding: utf-8

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