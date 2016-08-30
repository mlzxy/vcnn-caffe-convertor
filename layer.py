from util import *
from collections import defaultdict as ddict
from names import *


def dumpInputDimension(channels, width, height, fl=None):
    constant_int(fl, nChannels, channels, imgWidth, width, imgHeight, height)


def dumpMeanData(fl=None, fw = None,fd=None, data={}):
    if data['data']:
        code(fw,
"""
/*
The Mean Image data:
*/
""")
        float_array(file=fw,list=data['data'],name=MeanImage,
                shape=(data['channels']                 ,data["width"],data["height"]),
                decimal=DECIMAL_NUMBER)
        code(fd, extern(float_array(file=None,list=data['data'],name=MeanImage,
                shape=(data['channels']                 ,data["width"],data["height"]),
                decimal=DECIMAL_NUMBER, declare_only=True)))
    ## the dimension here will become the input for the function to be synthesized
    else:
        image_size = data['channels'] * data["width"] * data["height"]
        empty_list = image_size*[0]
        float_array(file=fw, list=empty_list, name=MeanImage,
                    shape=(data['channels'], data["width"] , data["height"]),
                    decimal=DECIMAL_NUMBER)
        code(fd, extern(float_array(file=None, list=empty_list, name=MeanImage,
                                    shape=(data['channels'], data["width"], data["height"]),
                                    decimal=DECIMAL_NUMBER, declare_only=True)))

    code(fl, """
#define MEAN_IMAGE_TYPE float
#define INPUT_IMAGE_TYPE  int
    """)




global layer_counter
layer_counter = 0
def dumpLayers(fl=None, fw=None, fd=None, layers=[]):
    global layer_counter
    len_l = len(layers)
    fl_string_list = []
    fw_string_list = []
    fl_string_list.append( "{0}  {1}[{2}] = {{".format(struct_layer_name,
                                     layers_varname,
                                     len_l+1))
    fw_string_list.append( "{0}  {1}[{2}] = {{".format(struct_layer_weight_name,
                                     layers_weight_varname,
                                     len_l))
    code(fd, extern("{0}  {1}[{2}];".format(struct_layer_weight_name,
                                     layers_weight_varname,
                                     len_l)))
    code(fd, extern("{0}  {1}[{2}];".format(struct_layer_name,
                                     layers_varname,
                                     len_l+1)))
    for ls in layers:
        s1,s2 =  dumpLayer(ls[0],ls[1])
        fl_string_list.append( s1)
        fw_string_list.append( s2)
        layer_counter += 1
        if layer_counter != len_l:
            fl_string_list.append( ',')
            fw_string_list.append( ',')
    fl_string_list.append(',')
    fl_string_list.append(empty_struct)
    fl_string_list.append( '};')
    fw_string_list.append( '};')
    for s in fl_string_list:
        code(fl, s)
    for s in fw_string_list:
        code(fw, s)






def dumpLayer(l, lproto):
    r = None
    global layer_counter
    type = l.ltype
    input_channel_num = None
    input_feature_map_height = None
    input_feature_map_width = None
    if len(l.shape) == 3:
        input_channel_num = l.shape[0]
        input_feature_map_height = l.shape[1]
        input_feature_map_width = l.shape[2]
    elif len(l.shape) == 1:
        input_channel_num = l.shape[0]
        input_feature_map_height = 1
        input_feature_map_width = 1
    else:
        raise 'Illigel data shape in layer'
    if type == CONVOLUTION:
        b0 = l.blobs[0]
        b1 = l.blobs[1]
        return createLayer(id=layer_counter,type=type,
                       conv_filter_channels=b0.channels,
                       conv_filter_size=b0.width,
                       conv_filter_num=b0.num,
                       conv_stride=l.stride,
                       conv_pad = l.pad,
                       input_channel_num=input_channel_num,
                       input_feature_map_height=input_feature_map_height,
                       input_feature_map_width=input_feature_map_width), \
               createWeight(
                id=layer_counter,
                conv_filter_weight=float_array_in_struct(list=b0.data, shape=b0.data.shape,
                                                          decimal=DECIMAL_NUMBER),
                conv_bias=float_array_in_struct(list=b1.data, shape=b1.data.shape,
                                                 decimal=DECIMAL_NUMBER)) ## problem in l.pad
    elif type == POOLING_AVG or type == POOLING_MAX:
        return createLayer(id=layer_counter,
                           type=type,
                           pl_kernel_size=l.kernel_size,
                           pl_stride=l.stride,
                           input_channel_num=input_channel_num,
                           input_feature_map_height=input_feature_map_height,
                           input_feature_map_width=input_feature_map_width), empty_struct
    elif type == INNER_PRODUCT:
        weight = l.blobs[0].data
        bias = l.blobs[1].data
        return createLayer( id=layer_counter,
                            type=type,
                            ip_channel_num=weight.shape[1],
                            ip_output_num=weight.shape[0],
                            input_channel_num=input_channel_num,
                            input_feature_map_height=input_feature_map_height,
                            input_feature_map_width=input_feature_map_width), \
               createWeight(id=layer_counter,
                            ip_weight=float_array_in_struct(
                                                    list=weight.tolist(),
                                                    shape=weight.shape,
                                                    decimal=DECIMAL_NUMBER),
                            ip_bias=float_array_in_struct(   ## problem here
                                                    list=bias.tolist(),
                                                    shape=bias.shape,
                                                    decimal=DECIMAL_NUMBER))
    elif type == SOFTMAX or type == RELU or type == RETURN_CALLBACK:
        return createLayer(id=layer_counter,
                           type=type,
                           input_channel_num=input_channel_num,
                           input_feature_map_height=input_feature_map_height,
                           input_feature_map_width=input_feature_map_width),empty_struct

    return r




