#!/usr/bin/env bash


echo "Generating Cifar10 Header"
python ./main.py ./data/cifar10/cifar10_quick_train_test.prototxt   ./data/cifar10/cifar10_quick_iter_5000.caffemodel.h5   ./data/cifar10/mean.binaryproto   ./output/cifar10/


echo "Generating Mnist Header"
python ./main.py  ./data/mnist/lenet_train_test.prototxt   ./data/mnist/lenet_iter_10000.caffemodel   no ./output/mnist/

echo "Generating Mnist Test Header"
python ./main.py ./data/mnist_test_nets/ip_only.prototxt ./data/mnist_test_nets/snapshots/ip_only_iter_10000.caffemodel   no ./output/test_mnist/ip_only/
python ./main.py ./data/mnist_test_nets/ip_conv.prototxt ./data/mnist_test_nets/snapshots/ip_conv_iter_10000.caffemodel   no ./output/test_mnist/ip_conv/
