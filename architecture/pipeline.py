#!/usr/bin/env python
# encoding: utf-8

from data_prepare.data_generate import *
from data_prepare.tools import *
from multiprocessing import Queue, Pipe, Manager, Value
from architecture.model_worker import InputModelWorker, EncoderModelWorker, DecoderModelWorker, OutputModelWorker

weights_path = "../checkpoint_list/checkpoint_UNet_5layers_200612_3092/UNet_5layers_200612_30902"
input_dir = "../dataset/caltechPedestrians/parallel_test"
output_dir = "../dataset/caltechPedestrians/parallel_test/output"


class Scheduler:
    def __init__(self,weights_path, output_dir, sleep_time, input_size):
        # load model
        self._weights_path = weights_path
        self._input_size = input_size
        self._output_dir = output_dir
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
                                                self._sleep_time))

    def start(self, test_imgs,name_imgs):
        # put all of images  into queue
        # for test_img in test_imgs:
        #     self._input_queue.put(test_img)
        # self._input_queue.put(None)
        for test_img in test_imgs:
            self._input_list.append(test_img)
        self._input_list.append(None)

        for name_img in name_imgs:
            self._name_list.append(name_img)
        print ("=" * 50)

        # start the workers
        for worker in self._workers:
            worker.start()

        # wait all workers finish
        for worker in self._workers:
            worker.join()


def run(weights_path, input_dir, output_dir, sleep_time):
    # load all image under the input_dir
    test_imgs, name_imgs = load_data_images(input_dir)

    H = test_imgs[0].shape[0]
    W = test_imgs[0].shape[1]
    C = test_imgs[0].shape[2]
    tile_height = H
    tile_width = W
    tile_size = 1

    data_string.append(len(test_imgs))
    data_string.append(sleep_time)
    data_string.append(tile_size)
    data_string.append(tile_height)
    data_string.append(tile_width)

    print("H:{} W:{} C:{}".format(H,W,C))
    # init scheduler
    x = Scheduler(weights_path, output_dir, sleep_time, input_size=(H,W,C))
    # start processing and wait for complete
    x.start(test_imgs,name_imgs)


if __name__ == '__main__':
    output_tile = "../dataset/caltechPedestrians/parallel_test/output_tile"
    xls_path = os.path.join(output_tile, "multi_runtime.xls")

    sleep_time = 0.1
    data_string = []
    start_time = time.time()
    run(weights_path, input_dir, output_dir, sleep_time)
    run_time = time.time() - start_time
    data_string.append(run_time)
    print("the run time is {}s".format(run_time))
    insert_worksheet(xls_path, data_string)






