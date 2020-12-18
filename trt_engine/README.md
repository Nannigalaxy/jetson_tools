# TensorRT engine exporter  
###Note: Following execution is to be performed on Jetson.   

## a. Using bash script  
This script exports model from keras h5 to Pb, ONNX and TRT engine format. These models are saved in exports/ directory.  

Give executable permission for bash script  
`$ chmod +x ./trt_engine_export`  
  
Usage:  
`./trt_engine_export /path/to/keras_model.h5 [batch_size, height, width, channel]`  

E.g. `./trt_engine_export ../keras/our_model_22_func.h5 [1,224,224,3]`  

## b. Manually without bash script
This procedure is written using example given in this [blog by nvidia](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/).  

To deploy deep learning on Jetson Nano trained keras model need to be convert to TensorRT engine. 
TensorRT is a high-performance neural network inference optimizer and runtime engine for production deployment. 

The workflow consists of the following steps:  
  
- Convert the TensorFlow/Keras model to a .pb file.  
- Convert the .pb file to the ONNX format.  
- Create a TensorRT engine. 
- Run inference from the TensorRT engine.  

## Keras model to .pb file  
Copy trained keras model to Jetson.   
./inference/converter/keras_to_pb.py script converts keras model to pb file.  
Input node and Ouput node are displayed at the end are to be noted for the next step.  
Usage: 
`
python3 converter/keras_to_pb.py --input cnn_model.h5 --output cnn_model.pb
`

## .pb file to ONNX
The second step is to convert the .pb model to the ONNX format. To do this, pip install tf2onnx.  
Specify input_layer static shape for dynamic shape for optimization profile within '[]' e.g. '[1,224,224,3]', [batch_size,height,width,channel].  
Usage:  
`
python3 -m tf2onnx.convert  --input cnn_model.pb --inputs input_layer:0[1,224,224,3] --outputs output_layer:0 --output cnn_model.onnx 
`
## Creating the TensorRT engine from ONNX  

Using built-in TensoRT library "trtexec" to validate onnx model and export engine model for inferencing. 
To know more available options/settings about trtexec, execute the following command.  
`/usr/src/tensorrt/bin/trtexec -h`

Example (FP32 to FP16):  
`/usr/src/tensorrt/bin/trtexec --onnx=cnn_model.onnx --fp16 --explicitBatch --useCudaGraph --workspace=1000 --saveEngine=cnn_model.engine`

