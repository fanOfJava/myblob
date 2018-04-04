# TextBoxes++: A Single-Shot Oriented Scene Text Detector

****
#Abstract
场景文字检测在文字识别中扮演中很重要的角色，和普通的物体检测不同的是，场景文字检测的难点主要在于，在自然图像中，文字的角度，大小以及长宽比各异。本篇论文提出了一种可以端到端训练的快速的检测任意角度文字的场景文字检测器。在各种公开数据集，包括ICDAR2015和COCO上做测试发现，检测器无论是速度还是精度都是棒棒哒！
****
#Introduction
场景文字检测的挑战主要包括：

1. various in both foreground text and background text
2. arbitrary oriented text
3. uncontrollable lighting conditions

传统的文字检测器按照原始检测物的不同可以分为：

1. 基于字符检测的
2. 基于word检测的
3. 基于text-line-based

按照检测的bounding boxes形状分可以分为：

1. Horizontal or near horizontal：这种方法很像一般的目标检测
2. Multi-oriented：这种方法变种比较多，具体可以参考文献[35]-[42]

本文介绍的TextBoxes++是一种word-based和Multi-oriented的检测器。TextBoxes++基于SSD，但由于传统的SSD对于极端长宽比的文字以及任意方向的文字检测都束手无策，所以作者就挖了个这个坑，提出了TextBoxes++，通过引入一个叫textboxes layer来解决这个问题。
****
#Overview
基于SSD，作者提出了一系列特殊的操作，使得网络能够检测不同方向的文字。

1. 提出了用四边形来表示任意方向的文字框(废话)
2. 将default-boxes回归到用四边形表示的bounding-boxes
3. 为了不仅保持横向default-boxes的密集性，而且还为了保证竖直方向的密集性，作者设置了default-boxes的竖直偏置
4. 作者在textboxes layer中设计了特殊的卷积(3*5)来替代传统的(3*3)卷积，以此来better handle text lines detecting
****
#Proposed network
整体网络结构如下图所示：

![](/img/textboxes++/network.png)
在VGG16的13层卷积的基础上，增加了10层卷积，以及6层textboxes layer（本质上还是卷积层），总共29层的网络，Fig2中介绍文本框用四边形或者ratated rectangle表示，本文经过测试发现，用四边形表示对于检测精度和召回更好。可以看到，TextBoxes++网络结构中除了卷积就是pooling，这样的话，对输入的图像大小没有限制，可以进行多尺度的训练和测试。所以对于某一个default-box，网络的输出包括：一个2维的概率(包含文字和背景概率)，一个4维的包含多边形的最小长方形box的，相对于default-box的偏差。在训练的时候，是根据这个box和default-box的IOU来区分哪个default-box负责ground-truth的预测。还有一个8维的四边形box的偏差量。所以textboxes layer的channel数量是2+4+8=14
