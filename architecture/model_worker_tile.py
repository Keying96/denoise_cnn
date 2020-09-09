#!/usr/bin/env python
# encoding: utf-8
from data_prepare.tiles import *
from multiprocessing import  Process

import numpy as np
import os
import time
import imageio

class InputModelWorker(Process):
    def __init__(self, weights_path, input_size,
                 input_conn, output_conn, model_id, start_flag):
        Process.__init__(self, name="ModelProcessor")
        self._weights_path = weights_path
        self._input_size  = input_size
        self._model_id = model_id
        self._input_conn = input_conn
        self._output_conn = output_conn
        self._start_flag = start_flag


    def run(self):
        from architecture.sub_model import load_sub_models
        sub_models = load_sub_models(self._weights_path,self._input_size)
        sub_model = sub_models[self._model_id]
        # sub_model.summary()

        flag = 0
        while True:
            if len(self._input_conn) == flag:
                print ("queue is empty")
                break

            if self._input_conn:
                # test_img = self._input_conn.get()
                test_img = self._input_conn[flag]

                # res_img = self.predict(sub_model, test_img)
                pre_tile_img_set = []
                for h in range(len(test_img)):
                    pre_tiles = []
                    for w in range(len(test_img[0])):
                        curr_tile = test_img[h][w]
                        pre_img = self.predict(sub_model, curr_tile)
                        pre_tiles.append(pre_img)
                    pre_tile_img_set.append(pre_tiles)

                self._output_conn.append(pre_tile_img_set)
                flag += 1
                self._start_flag.value += 1

                # print("{} ===== predict done".format(self._model_id))

    def predict(self, sub_model, test_img):

        test_img = np.expand_dims(test_img, axis=0)
        res_img = sub_model.predict(test_img)

        return  res_img


class EncoderModelWorker(Process):
    def __init__(self, weights_path, input_size,
                 input_conn, output_conn, model_id, start_flag, next_flag, sleep_time):
        Process.__init__(self, name="MiddleModelProcessor")
        self._weights_path = weights_path
        self._input_size = input_size
        self._model_id = model_id
        self._input_conn = input_conn
        self._output_conn = output_conn
        self._start_flag = start_flag
        self._next_flag = next_flag
        self._sleep_time = sleep_time

    def run(self):
        from architecture.sub_model import load_sub_models
        sub_models = load_sub_models(self._weights_path, self._input_size)
        sub_model = sub_models[self._model_id]
        # sub_model.summary()

        flag = 0
        while True:
            if len(self._input_conn) == flag and flag !=0:
                print ("{} conv is over".format(self._model_id))
                break

            if self._input_conn:
                # print("{} conv conn is not empty".format(self._model_id))
                # test_img = self._input_conn.get()
                test_img = self._input_conn[flag]

                # res_img = self.predict(sub_model, test_img)
                pre_tile_img_set = []
                for h in range(len(test_img)):
                    pre_tiles = []
                    for w in range(len(test_img[0])):
                        curr_tile = test_img[h][w]
                        pre_img = self.predict(sub_model, curr_tile)
                        pre_tiles.append(pre_img)
                    pre_tile_img_set.append(pre_tiles)

                self._output_conn.append(pre_tile_img_set)
                flag += 1
                self._start_flag.value -= 1
                self._next_flag.value += 1

            else:
                time.sleep(self._sleep_time)
                # print("{} conv is wating:{}".format(self._model_id, self._start_flag.value))




    def predict(self, sub_model, test_img):
        res_img = sub_model.predict(test_img)

        return  res_img


class DecoderModelWorker(Process):
    def __init__(self, weights_path, input_size,
                 input_conn, output_conn, crop_conn, model_id, start_flag, next_flag, sleep_time):
        Process.__init__(self, name="MiddleModelProcessor")
        self._weights_path = weights_path
        self._input_size = input_size
        self._model_id = model_id
        self._input_conn = input_conn
        self._output_conn = output_conn
        self._crop_conn = crop_conn
        self._start_flag = start_flag
        self._next_flag = next_flag
        self._sleep_time = sleep_time

    def run(self):
        from architecture.sub_model import load_sub_models
        sub_models = load_sub_models(self._weights_path, self._input_size)
        sub_model = sub_models[self._model_id]
        # sub_model.summary()

        flag = 0
        while True:
            if len(self._input_conn) == flag and flag !=0:
                print ("{} conv is over".format(self._model_id))
                break

            if self._start_flag.value:
                # print("{} conv conn is not empty".format(self._model_id))
                # test_img = self._input_conn.get()
                test_img = self._input_conn[flag]
                crop_part = self._crop_conn[flag]


                # res_img = self.predict(sub_model, test_img, crop_part)
                # print ("the shape of res img{}".format(res_img.shape))
                # self._output_conn.append(res_img)
                pre_tile_img_set = []
                for h in range(len(test_img)):
                    pre_tiles = []
                    for w in range(len(test_img[0])):
                        curr_tile = test_img[h][w]
                        crop_tile = crop_part[h][w]
                        pre_img = self.predict(sub_model, curr_tile, crop_tile)
                        pre_tiles.append(pre_img)
                    pre_tile_img_set.append(pre_tiles)

                self._output_conn.append(pre_tile_img_set)
                flag += 1
                self._start_flag.value -= 1
                self._next_flag.value += 1

            else:
                time.sleep(self._sleep_time)
                # print("{} conv is wating:{}".format(self._model_id, self._start_flag.value))


    def predict(self, sub_model, test_img, crop_part):
        res_img = sub_model.predict((test_img, crop_part))
        return  res_img


class OutputModelWorker(Process):
    def __init__(self, weights_path, input_size,
                 input_conn, output_conn, crop_conn,
                 model_id, start_flag, output_dir, name_list,overlap_px, sleep_time):
        Process.__init__(self, name="MiddleModelProcessor")
        self._weights_path = weights_path
        self._input_size = input_size
        self._model_id = model_id
        self._input_conn = input_conn
        self._output_conn = output_conn
        self._crop_conn = crop_conn
        self._start_flag = start_flag
        self._output_dir = output_dir
        self._name_list = name_list
        self._overlap_px = overlap_px
        self._sleep_time = sleep_time

    def run(self):
        from architecture.sub_model import load_sub_models
        sub_models = load_sub_models(self._weights_path, self._input_size)
        sub_model = sub_models[self._model_id]
        # sub_model.summary()

        flag = 0
        while True:
            if len(self._input_conn) == flag and flag !=0:
                print ("{} conv is over".format(self._model_id))
                break

            if self._start_flag.value:
                # test_img = self._input_conn.get()
                test_img = self._input_conn[flag]
                crop_part = self._crop_conn[flag]

                # res_img = self.predict(sub_model, test_img, crop_part)[0]
                pre_tile_img_set = []
                for h in range(len(test_img)):
                    pre_tiles = []
                    for w in range(len(test_img[0])):
                        curr_tile = test_img[h][w]
                        crop_tile = crop_part[h][w]
                        pre_img = self.predict(sub_model, curr_tile,crop_tile)[0]
                        pre_tiles.append(pre_img)
                    pre_tile_img_set.append(pre_tiles)

                res_img = self.tile_set2img(pre_tile_img_set, self._overlap_px)
                # print ("the shape of res img{}".format(res_img.shape))
                self.output_img(self._name_list[flag], res_img, self._output_dir)
                self._output_conn.append(res_img)
                flag += 1
                self._start_flag.value -= 1

            else:
                if flag != 0:
                    print ("{} conv is over".format(self._model_id))
                    break
                else:
                    time.sleep(self._sleep_time)
                    # print("{} conv is wating:{}".format(self._model_id, self._start_flag.value))


    def predict(self, sub_model, test_img, crop_part):
        res_img = sub_model.predict((test_img, crop_part))
        return  res_img

    # tile_set2img
    def tile_set2img(self, pre_tile_set, overlap_size):
        stitch_img = stitch_random_tiles(pre_tile_set, overlap_size)
        pre_img = np.minimum(np.maximum(stitch_img, 0), 255)
        return pre_img

    def output_img(self, name_img, res_img, output_dir):
        name_img = name_img.split(".")[0] + "_pre.jpg"
        output_dir = os.path.join(output_dir, name_img)
        # print(output_dir)

        imageio.imwrite(output_dir, res_img.astype("uint8"))
