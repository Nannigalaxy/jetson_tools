#!/bin/bash
# Written on Fri Dec 18 2020 By nannigalaxy

# HELP: 1st argument: Keras model path
#	2nd argument: Static input shape. [batch_size, height, width, channel] (optional)
# E.g. $ ./trt_engine_export /path/to/keras_model.h5 [1,224,224,3]


echo "	##############################################	"
echo "	TensorRT Engine model exporter for Jetson Nano	"
echo "	##############################################	"

# enable jetson_clocks
sudo jetson_clocks

# ******************  1	 *************************
# keras model to pb converter
model_path=$1
path=(`echo $model_path | tr "." "\n"`)
out_name=(`echo ${path[0]} | tr "/" "\n"`)
out_model=${out_name[-1]}
python3 ./converter/keras_to_pb.py --i $model_path --o $out_model.pb

# read input and output layer from temp file
file="/tmp/tmp_layers"
raw_str=$(cat "$file")
layers=(`echo $raw_str | tr "," "\n"`)

input_l=${layers[0]}
output_l=${layers[1]}

echo -e "input layer:"$input_l "\noutput layer:"$output_l

# ******************  2	 *************************
# pb to onnx converter
python3 -m tf2onnx.convert  --input $out_model.pb --inputs $input_l:0$2 --outputs $output_l:0 --output $out_model.onnx

# ******************  3	 *************************
# onnx to engine converter
/usr/src/tensorrt/bin/trtexec --onnx=$out_model.onnx --fp16 --explicitBatch --useCudaGraph --workspace=1000 --saveEngine=${out_model}_16.engine
/usr/src/tensorrt/bin/trtexec --onnx=$out_model.onnx --explicitBatch --useCudaGraph --workspace=1000 --saveEngine=$out_model.engine

# ******************  4	 *************************
mkdir -p exports/{pb,onnx,engine,in_out_nodes} 

# Export converted models 
echo -e "input layer:" $input_l "\noutput layer:" $output_l > "./exports/in_out_nodes/"$out_model.txt
mv ./$out_model.pb ./exports/pb/
mv ./$out_model.onnx ./exports/onnx/
mv ./${out_model}_16.engine ./exports/engine/
mv ./$out_model.engine ./exports/engine/

rm -rf /tmp/tmp_layers
