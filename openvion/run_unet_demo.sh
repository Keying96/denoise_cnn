if [ -d "openvino" ]; then
  cd openvino
fi

OPENVINO_LIB=${INTEL_OPENVINO_DIR}/inference_engine/lib/intel64/

python3 inference_openvino.py -l ${OPENVINO_LIB}/libcpu_extension_avx512.so \
       --plot