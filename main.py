#!/usr/bin/python
import caffe_pb2 as caffe_proto
from google.protobuf.text_format import Merge
import caffe
from layer import *
import os
from easydict import EasyDict as edict




#
# def clc():
#     os.system('cls' if os.name == 'nt' else 'clear')


def readMeanFile(mean_path=''):
    with open(mean_path, 'rb') as f:
        blob = caffe_proto.BlobProto()
        blob.ParseFromString(f.read())
        f.close()
        return {
            'data':blob.data._values,
            # 'channels':blob.channels,
            # 'width':blob.width,
            # 'height':blob.height
         }



def readCaffeModelFile(txt_path='', binary_path=''):
    net = caffe.Net(txt_path, binary_path, caffe.TEST)
    result = []
    net_proto = caffe_proto.NetParameter()
    with open(txt_path,'r') as f:
        Merge(f.read(), net_proto)

    dimension = net.blobs['data'].data.shape[1:]
    proto_layer_counter = 0
    scale = 1.0
    for i in range(len(net.layers)):
        l = net.layers[i]
        type = l.type.upper()
        if type not in layer_not_in_prototxt:
            proto_layer_counter+=1
        lpt = net_proto.layer[proto_layer_counter]
        if layer_skipped[type]:
            if type == DATA:
                scale = lpt.transform_param.scale
            continue
        else:
            l.ltype = type
            if type == POOLING:
                p = lpt.pooling_param
                fixPooling(l,p)
            elif type == SOFTMAX_WITHLOSS or type == SOFTMAX:
                l.ltype = SOFTMAX
            elif type == CONVOLUTION:
                if len(lpt.convolution_param.stride._values) == 0:
                    l.stride = 0
                else:
                    l.stride = lpt.convolution_param.stride._values[0]
                if len(lpt.convolution_param.pad._values) == 0:
                    l.pad = 0
                else:
                    l.pad = lpt.convolution_param.pad._values[0]

            # dimension = None
            l.shape = dimension
            if net.blobs.has_key(lpt.name):  ## because the shape is its input dimension, which is contained in last layer
                if len(net.blobs[lpt.name].data.shape) != 0:
                    dimension = net.blobs[lpt.name].data.shape[1:]

            result.append([l, lpt])

## use the name in the net_proto.layer to query the input_dimension and set
## if the last input_dimension is not null, set it.
## so even if relu can't search it, it still gets the correct result
    result.append([edict({
            'ltype': RETURN_CALLBACK,
            'shape': dimension
    }), None])
    dimension = net.blobs['data'].data.shape[1:]

    return result, dimension, scale


def main(mean_path='',txt_path='',binary_path='', output_dir=os.getcwd()):
    if mean_path:
        mean_data = readMeanFile(mean_path=mean_path)
        print("Readed in  {0}".format(mean_path))
    else:
        mean_data = {
            'data':None
        }
    layer_data, dimension, scale= readCaffeModelFile(txt_path=txt_path, binary_path=binary_path)
    channels = dimension[0]
    height = dimension[1]
    width=dimension[2]
    # clc()

    print("           {0}".format(txt_path))
    print("           {0}".format(binary_path))
    with open(os.path.join(output_dir, default_layer_filename),'w') as fl:
        with open(os.path.join(output_dir, default_weight_filename),'w') as fw:
            print("Transforming them into c-struct.")
            ifdef(fl, include_guard_layer_def)
            # ifdef(fw, include_guard_layer_weight)
            code(fw, "#include \"{0}\"".format(default_layer_filename))
            code(fl, "static int const {0} = {1};".format(nLayers, len(layer_data)))
            code(fl, "static float const {0}= {1};".format(dataScale, scale))
            code(fl, "static int const {0} = {1};".format(nOutput, layer_data[len(layer_data)-2][0].shape[0]))
            dumpLayerDefinition(fl)
            dumpWeightDefinition(fl)
            dumpInputDimension(channels, width, height, fl=fl)
            constant_int(fl, nLayerTypes, len(layer_type_list))
            mean_data['channels'] = channels
            mean_data['width'] = width
            mean_data['height'] = height
            dumpMeanData(fl=fl, fw=fw,fd=fl,data=mean_data)
            dumpLayers(fl=fw,fw=fw, fd=fl, layers=layer_data)

            endif(fl, include_guard_layer_def)
            # endif(fw, include_guard_layer_weight)
            print('Output data into c-header files, in the directory {0}.'.format(output_dir))
            print('Finished')



import sys
if __name__ == "__main__":
    ag = sys.argv
    mean_path = ag[3]
    if mean_path == "no":
        mean_path = None
    else:
        mean_path = os.path.abspath(mean_path)
    main(mean_path=mean_path,
         txt_path=os.path.abspath(ag[1]),
         binary_path=os.path.abspath(ag[2]),
         output_dir=os.path.abspath(ag[4]))
