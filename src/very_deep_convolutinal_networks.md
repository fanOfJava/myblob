# VERY DEEP CONVOLUTIONAL NETWORKS


## Abstract

本篇论文我们主要探究卷积神经网络的深度对于大规模图像识别准确性的作用。本文主要的贡献在于通过增加大量使用3乘3卷积的网络层数（16-19层），来达到相比以前工作性能的大幅提高。

## Introduction
卷积神经网络近年来在大规模图像和视频识别中取得了巨大的成功，这归功于大规模图片数据集的公开以及高性能的计算系统的开发。

随着卷积神经网络在计算机视觉领域应用的增多，人们做了很多尝试，去提高原始AlexNet的性能。和以前的尝试不同，本论文中我们尝试着增加网络的***层数***来提高神经网络的性能。

## Convnet Configurations
为了说明网络层数的增加对性能的影响，我们的卷积网络的设计遵照alexnet的设计准则

### Architecture
在训练中，卷积神经网络的输入大小为224 × 224 RGB图片，唯一做的图片预处理是减掉均值(分别计算图片集上的每一个像素的均值)。预处理之后的图片被送入卷积层，我们使用非常小的3 × 3卷积核(能捕捉前后左右信息的最小单位)。我们也用1 × 1卷积核，他可以看做是对输入channels做线性变换。卷积的stride为1，padding也为1.max-pool仅在指定的卷积层之后做，2 × 2 pixel window, with stride 2。

一系列卷积层之后是三层全连接层，前面两层全连接层都是4096个channles，最后一层全连接层由于是需要分1000类，所以为1000个channels，第三层全连接层后接softmax层。

所有的隐藏层用的激活函数都为ReLU，在本文中我们不用Local Responose Normalisation，因为我们通过实验发现LRN并没有提高模型的性能，反而徒增gpu显存占用和计算时间。

### Configurations
卷积层的设计入表1所示
![](../img/very_deep_convolutional_networks/table_1.png)
