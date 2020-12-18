# https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/

import argparse
import tensorflow.compat.v1 as tf
# https://github.com/tensorflow/tensorflow/issues/3986#issuecomment-568618982
tf.disable_eager_execution()
from tensorflow.keras.models import load_model
# import keras.backend as K
tf.keras.backend.set_learning_phase(0)

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str)
parser.add_argument('--output',type=str)
opt = parser.parse_args()

print("Input Keras:", opt.input)
print("Output pb file:", opt.output)

def keras_to_pb(model, output_filename, output_node_names):

   """
   This is the function to convert the Keras model to pb.

   Args:
      model: The Keras model.
      output_filename: The output .pb file name.
      output_node_names: The output nodes of the network. If None, then
      the function gets the last layer name as the output node.
   """

   # Get the names of the input and output nodes.
   # for i in range(len(model.layers)):
   #  print(model.layers[i].get_output_at(0).name.split(':')[0])
   in_name = model.layers[0].get_output_at(0).name.split(':')[0]

   if output_node_names is None:
       output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]
      
#   for i in range(len(model.layers)):
#    print([model.layers[i].get_output_at(0).name.split(':')[0]])
#   print(model.output.op.name)
   sess = tf.keras.backend.get_session()

   # The TensorFlow freeze_graph expects a comma-separated string of output node names.
   output_node_names_tf = ','.join(output_node_names)
   no = [n.name for n in tf.get_default_graph().as_graph_def().node]
#   print("nodes",no)
   frozen_graph_def = tf.graph_util.convert_variables_to_constants(
       sess,
       sess.graph_def,
       output_node_names)

   sess.close()
   wkdir = ''
   tf.train.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)

   return in_name, output_node_names

keras_model = opt.input
model = load_model(keras_model)

# Convert the Keras model to a .pb file
in_tensor_name, out_tensor_names = keras_to_pb(model, opt.output, None) 

#print(in_tensor_name, out_tensor_names)
out = in_tensor_name +","+ out_tensor_names[0]
with open("/tmp/tmp_layers","w") as f:
   f.write(out)

# Next convert tf to onnx
# python -m tf2onnx.convert  --input cnn_model.pb --inputs conv2d/Relu:0 --outputs dense_1/Softmax:0 --output cnn3.onnx 
