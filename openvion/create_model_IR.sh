FROZEN_MODEL="../frozen_model/saved_model_frozen.pb"
INTEL_OPENVINO_DIR "/opt/intel/openvino/"

if [ ! -f $FROZEN_MODEL ]; then
   echo "File $FROZEN_MODEL doesn't exist."
   echo "Please make sure you have a trained model and then run the script: "
   echo "'python helper_scripts/convert_keras_to_tensorflow_serving_model.py --input_filename output/unet_model_for_decathlon.hdf5'"
   echo "The directions at the end of the script will show you the commands to"
   echo "create a frozen model."
   exit 1
fi

# For CPU
python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py \
      --input_model $FROZEN_MODEL \
      --input_shape=[1,321,481,3] \
      --data_type FP16  \
      --output_dir models/FP32  \
      --model_name unet5_saved_model