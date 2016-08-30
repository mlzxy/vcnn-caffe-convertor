# Caffe Convertor

Convert a caffe model to a C struct array.


## Install


1.
```shell
[sudo] pip install easydict protobuf
```

2. Install Caffe and pycaffe https://github.com/BVLC/caffe


## Example Usage

```shell
chmod +x ./generate.sh
./generate.sh  
```

will generate
1. google lenet for mnist
2. caffe cifar10 quick train network.

Look into `generate.sh` for how to use the commandline python tool.



## Output

**caffe_model_layer.h** and **caffe_model_layer.cpp** will be generated.


- [caffe_model_layer.h](https://bitbucket.org/xyz0/vcnn-caffe-convertor/src/ac6cbb20e669a383092049fe81e0d69b35dfc6b9/output/mnist/caffe_model_layer.h?at=master&fileviewer=file-view-default) contains layer-definition, layer-weight definition.

-  [caffe_model_layer.cpp](https://bitbucket.org/xyz0/vcnn-caffe-convertor/raw/ac6cbb20e669a383092049fe81e0d69b35dfc6b9/output/mnist/caffe_model_weight.h) contains layer-weight data, layer structure and image mean (a huge file.)


- These two files must be put into same folder.



- In the FPGA, the weight is huge and may not fit into BRAM/DRAM. So I make the definition and weight data seperate.
