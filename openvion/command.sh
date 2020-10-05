python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf --data_type FP16 --reverse_input_channels --input_shape [1,299,299,3] --input input --mean_values input[127.5,127.5,127.5] --scale_values input[127.50000414375013] --output InceptionV3/Predictions/Softmax --input_model /home/zhu/PycharmProjects/denoise_cnn/openvion/frozen_models/inception_v3_2016_08_28_frozen.pb
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py   --data_type FP16 --reverse_input_channels --input_shape [1,321,481,3] --input_model /home/zhu/PycharmProjects/denoise_cnn/openvion/frozen_models/simple_frozen_graph.pb --model_name unet5_save_model --output Identity --output_dir /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924

python3 /home/zhu/PycharmProjects/denoise_cnn/openvion/unet_denoise_python.py  -m /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_1/unet5_save_model.xml -i /home/zhu/PycharmProjects/denoise_cnn/dataset/caltechPedestrians/parallel_test/noise_img/94079_noise.jpg --cpu_extension /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924/libcpu_extension.so


source /opt/intel/openvino/bin/setupvars.sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf   --data_type FP16 --input_shape [1,321,481,3] --saved_model_dir /home/zhu/PycharmProjects/denoise_cnn/openvion/UNet_5layers_20-09-11-15-55  --tensorboard_logdir  /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_1 --output_dir /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_1
python3 /home/zhu/PycharmProjects/denoise_cnn/openvion/unet_denoise_v4_python.py  -m /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_1/saved_model.xml -i /home/zhu/PycharmProjects/denoise_cnn/dataset/caltechPedestrians/parallel_test/noise_img/94079_noise.jpg

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf   --data_type FP16  --input_shape [1,321,481,3] --saved_model_dir /home/zhu/PycharmProjects/denoise_cnn/openvion/UNet_5layers_200612_30902 --tensorboard_logdir  /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_2 --output_dir /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_2
python3 /home/zhu/PycharmProjects/denoise_cnn/openvion/unet_denoise_v4_python.py  -m /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_2/saved_model.xml -i /home/zhu/PycharmProjects/denoise_cnn/dataset/caltechPedestrians/parallel_test/noise_img/94079_noise.jpg
python3 /home/zhu/PycharmProjects/denoise_cnn/openvion/unet_denoise_v4_python.py  -m /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_2/saved_model.xml -i /home/zhu/PycharmProjects/denoise_cnn/dataset/caltechPedestrians/parallel_test/noise_img/97017_noise.jpg

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf   --data_type FP16  --reverse_input_channels --input_shape [1,321,481,3] --saved_model_dir /home/zhu/PycharmProjects/denoise_cnn/openvion/UNet_5layers_200612_30902 --tensorboard_logdir  /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3 --output_dir /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3
python3 /home/zhu/PycharmProjects/denoise_cnn/openvion/unet_denoise_v4_python.py  -m /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/saved_model.xml -i /home/zhu/PycharmProjects/denoise_cnn/dataset/caltechPedestrians/parallel_test/noise_img/97017_noise.jpg
python3 /home/zhu/PycharmProjects/denoise_cnn/openvion/unet_denoise_v4_python.py  -m /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/saved_model.xml -i /home/zhu/PycharmProjects/denoise_cnn/dataset/caltechPedestrians/parallel_test/noise_img/94079_noise.jpg

docker rm -f /ubuntu-openvino2020R3

docker run --device /dev/dri \
--device=/dev/ion:/dev/ion \
-v /var/tmp:/var/tmp \
--name ubuntu-openvino2020R3 \
-d ubuntu:openvino_2020R3 \
/bin/sh
/bin/bash
/sbin/init
/bin/sh

docker: Error response from daemon: OCI runtime create failed:
container_linux.go:348: starting container process caused "exec: \"/sbin/init\": stat /sbin/init: no such file or directory": unknown.

docker run  --device /dev/dri \
-it \
--device=/dev/ion:/dev/ion \
-v /var/tmp:/var/tmp \
--name ubuntu-openvino2020R3 \
-d ubuntu:openvino_2020R3 \
/bin/bash

docker exec -it ubuntu-openvino2020R3 /bin/bash
source /opt/intel/openvino/bin/setupvars.sh
cd /root
chmod 755 ./*
export LD_LIBRARY_PATH=/root:$LD_LIBRARY_PATH
./unet_denoise_v4_python.py -i ./94079_noise.jpg  -m ./saved_model.xml
unet_denoise_v4_python.py -i 94079_noise.jpg  -m saved_model.xml
./unet_denoise_v4_python.py -i 94079_noise.jpg  -m saved_model.xml
python3 unet_denoise_v4_python.py -i 94079_noise.jpg  -m  saved_model.xml

docker cp /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/94079_noise.jpg  ubuntu-openvino2020R3:/root/

:set ff=unix
:wq


create model_IR
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf   --data_type FP16  --reverse_input_channels --input_shape [1,321,481,3] --saved_model_dir /home/zhu/PycharmProjects/denoise_cnn/openvion/UNet_5layers_200612_30902 --tensorboard_logdir  /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20201001 --output_dir /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20201001

run unet.py

docker run -it --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb --name ubuntu-openvino2020.4 -d openvino/ubuntu18_dev:2020.4 /bin/bash
docker exec -it ubuntu-openvino2020.4 /bin/bash
cd /home/openvino
docker cp /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/94079_noise.jpg  ubuntu-openvino2020.4:/home/openvino
docker cp /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/saved_model.bin  ubuntu-openvino2020.4:/home/openvino
docker cp /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/saved_model.xml  ubuntu-openvino2020.4:/home/openvino
docker cp /home/zhu/PycharmProjects/denoise_cnn/openvion/unet_denoise_v4_python.py  ubuntu-openvino2020.4:/home/openvino

docker save -o /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/ubuntu18-openvino2020.4.tar openvino/ubuntu18_dev:2020.4
docker save -o /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/ubuntu18-openvino2020.4.tar ubuntu:openvino_2020R3
docker save -o /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/ubuntu18-openvino2020.4.tar ubuntu_openvino:2020.4

docker build . -t ubuntu:openvino_2020R3

docker ps
# export!!!!
docker export 79682725b315    > /home/zhu/PycharmProjects/denoise_cnn/openvion/model_IR/20200924_3/ubuntu-openvino.tar

cat ubuntu-openvino.tar | docker import - ubuntu-openvino
docker run  --device /dev/dri \
-it \
--device=/dev/ion:/dev/ion \
-v /var/tmp:/var/tmp \
--name ubuntu-openvino2020.4 \
-d ubuntu-openvino:latest \
/bin/bash

docker exec -it ubuntu-openvino2020.4 /bin/bash
source /opt/intel/openvino/bin/setupvars.sh

docker run  --device /dev/dri \
-it \
--device=/dev/ion:/dev/ion \
-v /var/tmp:/var/tmp \
--name ubuntu-openvino2020R4 \
-d  0d97736343fe \
/bin/bash
docker exec -it ubuntu-openvino2020R4 /bin/bash

docker build . -t ubuntu-openvino:2020R4

docker run -it --name ubuntu-openvino2020R4 -d ubuntu-openvino:2020R4 /bin/bash
docker exec -it ubuntu-openvino2020R4 /bin/bash
docker run  --device /dev/dri \
-it \
--device=/dev/ion:/dev/ion \
-v /var/tmp:/var/tmp \
--name ubuntu-openvino2020R4 \
-d  19eb3dd40e1b  \
/bin/bash