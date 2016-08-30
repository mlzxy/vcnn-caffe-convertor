from collections import defaultdict as ddict
from util import *

DATA = 'DATA'
layer_skipped_list = ['Data'.upper(),
                      'Split'.upper(),
                      'Accuracy'.upper()]

layer_not_in_prototxt = ['Split'.upper()]


default_layer_filename = 'caffe_model_layer.h'
default_weight_filename = 'caffe_model_layer.cpp'

layer_skipped = ddict(lambda: False)
for k in layer_skipped_list:
    layer_skipped[k] = True

POOLING = 'Pooling'.upper()
SOFTMAX_WITHLOSS = 'SoftmaxWithLoss'.upper()

## all the layers needed to be implemented
CONVOLUTION = 'Convolution'.upper()
POOLING_AVG = 'Pooling_avg'.upper()
POOLING_MAX = 'Pooling_max'.upper()
RELU = 'ReLU'.upper()
INNER_PRODUCT = 'InnerProduct'.upper()
SOFTMAX = 'Softmax'.upper()
RETURN_CALLBACK = 'RETURN_CALLBACK'

layer_type_list = [
    CONVOLUTION,
    POOLING_AVG,
    POOLING_MAX,
    RELU,
    INNER_PRODUCT,
    SOFTMAX,
    RETURN_CALLBACK
]


DECIMAL_NUMBER = 6
nLayerTypes = 'nLayerTypes'
nChannels = 'nChannels'
imgWidth = 'imgWidth'
nOutput = 'nOutput'
imgHeight = 'imgHeight'
MeanImage = 'mean_image'
nLayers = 'nLayers'

struct_layer_type_name = 'LayerType'
struct_layer_name = 'Layer'
struct_layer_weight_name = 'LayerWeight'
layers_varname = 'layers'
layers_weight_varname = 'layers_weight'


## Convolution
conv_filter_weight = 'conv_filter_weight'
conv_bias = 'conv_bias'
conv_filter_channels = 'conv_filter_channels'
conv_filter_size = 'conv_filter_size'
conv_filter_num = 'conv_filter_num'
conv_stride = 'conv_stride'
conv_pad = 'conv_pad'

##pooling
pl_kernel_size = 'pl_kernel_size'
pl_stride = 'pl_stride'

##inner product
ip_weight = 'ip_weight'
ip_bias = 'ip_bias'
ip_channel_num = 'ip_channel_num'
ip_output_num = 'ip_output_num'

input_channel_num = 'input_channel_num'
input_feature_map_height = 'input_feature_map_height'
input_feature_map_width = 'input_feature_map_width'
dataScale = 'dataScale'
empty_struct = '{}'
one_array = '(float [1]){0}'

include_guard_layer_def = 'CAFFE_LAYER_DEF'
include_guard_layer_weight = 'CAFFE_LAYER_WEIGHT'

def ifdef(fl, gd):
    code(fl, "#ifndef {0}_H\n#define {0}_H".format(gd))

def endif(fl, gd):
    code(fl, "#endif /* {0}_H */".format(gd))


def dumpLayerDefinition(fl):
    code(fl, 'typedef enum {{{0}}} {1};'.format(','.join(layer_type_list), struct_layer_type_name),
         """
typedef  struct {{
    int id;
    {1} type;
    /*
     # Meta Data
    All meta data for every type of layer are put here, for easy initialization. HLS C don't allow dynamic memory operation.
    Hopefully the complier will optimize the unused variable for us.(It should be able to.)
    */
    /* for convolutional layer
     the conv layer here won't take care about padding, since we gonna use pixel streaming,
     FPGA is best for streaming processing, fast and efficient*/

    int {2};
    int {3};
    int {4};
    int {5};
    int {6};

    /*for pooling layer*/
    int {7};
    int {8};

    /*for inner product layer*/
    int {9};
    int {10};


    int {11};
    int {12};
    int {13};

    float* input_data;
}} {0};


         """.format(struct_layer_name, struct_layer_type_name,
                       ## conv_filter_weight,
                       ## conv_bias,
                        conv_filter_channels,
                        conv_filter_size,
                        conv_filter_num,
                        conv_stride,
                        conv_pad,

                        pl_kernel_size,
                        pl_stride,

                        ip_channel_num,
                        ip_output_num,
                        ## input
                        input_channel_num,
                        input_feature_map_height,
                        input_feature_map_width
                    ))

def createLayer(id=0, type=CONVOLUTION, conv_filter_channels=0, conv_filter_size=0, conv_filter_num=0, conv_stride=0, conv_pad = 0,
                pl_kernel_size=0, pl_stride=0, ip_channel_num=0, ip_output_num=0,
                input_channel_num=0, input_feature_map_height=0, input_feature_map_width=0):
    feature_map_size = input_channel_num* input_feature_map_height*input_feature_map_width
    return """
{{
{0},
{1},
{2},
{3},
{4},
{5},
{6},
{7},
{8},
{9},
{10},
{11},
{12},
{13},
{14}
}}
""".format(id, type, conv_filter_channels, conv_filter_size, conv_filter_num, conv_stride, conv_pad,
           pl_kernel_size, pl_stride, ip_channel_num, ip_output_num,
           input_channel_num, input_feature_map_height, input_feature_map_width, '(float [{0}]){1}'.format(feature_map_size,listToCArrayString([0]*feature_map_size)))



def dumpWeightDefinition(fl):
    code(fl,
"""
#ifndef LAYER_WEIGHT_STRUCT
#define LAYER_WEIGHT_STRUCT
typedef struct {{

  int id;
  //for convolutional layer
  float *{0}; // 32->3->5x5
  float *{1}; //1x32

  /*for inner product layer*/
  float *{2};
  float *{3};

}} {4};
#endif
""".format(conv_filter_weight,conv_bias,ip_weight,ip_bias,struct_layer_weight_name))


def createWeight(id=0, conv_filter_weight=one_array, conv_bias=one_array, ip_weight=one_array, ip_bias=one_array):
    return """
{{
{0}, //id
{1}, //conv_filter_weight
{2}, //conv_bias
{3}, //ip_weight
{4}  //ip_bias
}}
""".format(id, conv_filter_weight, conv_bias, ip_weight, ip_bias)







def fixPooling(l, p):
    if p.pool == p.AVE:
        l.ltype = POOLING_AVG
    elif p.pool == p.MAX:
        l.ltype = POOLING_MAX
    l.stride = p.stride
    l.kernel_size = p.kernel_size
