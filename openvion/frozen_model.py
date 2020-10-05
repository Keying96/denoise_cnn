#!/usr/bin/env python
# encoding: utf-8

from cnn.model_5layers import *

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def main():
    model, model_name = unet()

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, dtype=tf.float32))

    # Get frozen ConcreteFunction
    # frozen_func = tf.graph_util.convert_variables_to_constants(full_model)
    frozen_func =  convert_variables_to_constants_v2(full_model)

    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="simple_frozen_graph.pb",
                      as_text=False)


    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_models/simple_frozen_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

if __name__ == "__main__":

    main()