# TextBoxes++: A Single-Shot Oriented Scene Text Detector

****
#Abstract
场景文字检测在文字识别中扮演中很重要的角色，和普通的物体检测不同的是，场景文字检测的难点主要在于，在自然图像中，文字的角度，大小以及长宽比各异。本篇论文提出了一种可以端到端训练的快速的检测任意角度文字的场景文字检测器。在各种公开数据集，包括ICDAR2015和COCO上做测试发现，检测器无论是速度还是精度都是棒棒哒！

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

1. Horizontal or near horizontal
2. Multi-oriented

本文介绍的textboxes++是一种word-based和Multi-oriented的检测器。



