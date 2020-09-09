#!/usr/bin/env python
# encoding: utf-8

from data_prepare.data_generate import *
from data_prepare.tiles import *
from multiprocessing import  Manager, Value
from architecture.model_worker_tile import InputModelWorker, EncoderModelWorker, DecoderModelWorker, OutputModelWorker

import time

weights_path = "../checkpoint_list/checkpoint_UNet_5layers_200612_3092/UNet_5layers_200612_30902"
input_dir = "../dataset/caltechPedestrians/parallel_test"
# output_dir = "../dataset/caltechPedestrians/parallel_test/output"
output_dir = "../dataset/caltechPedestrians/parallel_test/output_tile"
xls_path = os.path.join(output_dir, "multi_runtime.xls")


class Scheduler:
    def __init__(self,weights_path, output_dir, overlap_px, tile_size, sleep_time, input_size):
        # load model
        self._weights_path = weights_path
        self._input_size = input_size
        self._output_dir = output_dir
        self._overlap_px = overlap_px
        self._tile_size = tile_size
        self._sleep_time = sleep_time

        # the queue of load images
        self._name_list = Manager().list()
        self._input_list = Manager().list()
        self._conv1_output = Manager().list()
        self._conv2_output = Manager().list()
        self._conv3_output = Manager().list()
        self._conv8_output = Manager().list()
        self._output_list = Manager().list()

        # flag
        # self._start_conv2 = 0
        self._start_conv2 = Value("d", 0)
        self._start_conv3 = Value("d", 0)
        self._start_conv8 = Value("d", 0)
        self._start_output = Value("d", 0)

        # init process
        self._init_workers()

    def _init_workers(self):
        # sub_models = load_sub_models(self._unet_model)
        self._id_input = 0
        self._id_conv2 = 1
        self._id_conv3 = 2
        self._id_conv8 = 3
        self._id_output = 4

        self._workers = list()
        # input
        self._workers.append(InputModelWorker( self._weights_path , self._input_size,
                                          self._input_list, self._conv1_output,
                                               self._id_input, self._start_conv2))
        # conv2
        self._workers.append(EncoderModelWorker( self._weights_path , self._input_size,
                                          self._conv1_output, self._conv2_output,
                                                 self._id_conv2, self._start_conv2, self._start_conv3,
                                                 self._sleep_time))

        # conv3
        self._workers.append(EncoderModelWorker( self._weights_path , self._input_size,
                                          self._conv2_output, self._conv3_output,
                                                 self._id_conv3, self._start_conv3, self._start_conv8,
                                                 self._sleep_time))

        #conv8
        self._workers.append(DecoderModelWorker( self._weights_path , self._input_size,
                                          self._conv3_output, self._conv8_output, self._conv2_output,
                                                 self._id_conv8, self._start_conv8, self._start_output,
                                                 self._sleep_time))

        #output
        self._workers.append(OutputModelWorker( self._weights_path , self._input_size,
                                          self._conv8_output, self._output_list, self._conv1_output,
                                                 self._id_output, self._start_output, self._output_dir, self._name_list,
                                                self._overlap_px, self._sleep_time))

    def start(self, test_imgs,name_imgs):

        for test_img in test_imgs:
            # self._input_list.append(test_img)
            test_img_set = img2tile_set(test_img, self._overlap_px, self._tile_size)
            self._input_list.append(test_img_set)
        # self._input_list.append(None)

        for name_img in name_imgs:
            self._name_list.append(name_img)
        print ("=" * 50)

        # start the workers
        for worker in self._workers:
            worker.start()

        # wait all workers finish
        for worker in self._workers:
            worker.join()


# img2tile_set
def img2tile_set(noise_img, overlap_size, tile_size):
    overlap_noise_img = overlap_tile(overlap_size, noise_img)
    tile_img_set = split_images2_random_pathes(overlap_noise_img, overlap_size, tile_size)
    return tile_img_set

def get_img_HWC(gt_patch,overlap_size,tile_size):
    img = gt_patch[0]
    tile_img_set = img2tile_set(img, overlap_size, tile_size)
    H = tile_img_set[0][0].shape[0]
    W = tile_img_set[0][0].shape[1]
    C = tile_img_set[0][0].shape[2]
    return H,W,C

def run(weights_path, input_dir, output_dir, overlap_px, tile_size, sleep_time,data_string):
    # load all image under the input_dir
    gt_imgs, name_imgs = load_data_images(input_dir)
    test_imgs = poisson_noise_imgs(gt_imgs,lam_noise)
    H,W,C = get_img_HWC(test_imgs, overlap_px, tile_size)
    tile_height = H
    tile_width = W
    print("H:{} W:{} C:{}".format(H,W,C))

    data_string.append(len(test_imgs))
    data_string.append(sleep_time)
    data_string.append(tile_size)
    data_string.append(tile_height)
    data_string.append(tile_width)

    # init scheduler
    x = Scheduler(weights_path, output_dir, overlap_px, tile_size, sleep_time, input_size=(H,W,C))
    # start processing and wait for complete
    x.start(test_imgs,name_imgs)


if __name__ == '__main__':
    lam_noise = 20
    overlap_px = 4
    sleep_time = 0.1
    # tile_size = 2
    # tile_size_set = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    tile_size_set = [15, 20, 25, 30, 35]


    # ImgNum, SleepTime, TileSize, Runtime
    for tile_size in tile_size_set:
        data_string = []
        start_time = time.time()
        run(weights_path, input_dir, output_dir, overlap_px, tile_size, sleep_time, data_string)
        run_time = time.time() -start_time
        data_string.append(run_time)
        print("the run time is {}s".format(run_time))
        insert_worksheet(xls_path,data_string)






