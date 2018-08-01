# Detecting Text in Natural Image with Connectionist Text Proposal Network

****
# Abstract
场景文字端到端的识别，主流方法都分为两部，即场景文字的检测和识别。作为识别的基础模块，场景文字检测也直是研究的热点，CTPN是发表在2016-ECCV中的一种场景文本算法，在水平文本检测方面性能较好，如下图所示：

![](/img/ctpn/detecting_result1.png)
****
# Introduction
传统的文字检测方法，有着自己的局限性，不是检测不够准确就是后处理非常复杂，没法做到实时可用。随着深度学习的发展，通用物体的检测已经达到了较高的准确率以及较快的速度，类似Faster-RCNN等。但是直接将通用物体检测的模型用来做通用文字检测会遇到如下问题：

1. 通用物体检测中，检测物体有一个明确的边界，而文本没有。
2. 通用物体检测中，检测准确率要求比较低，检测框和ground-truth之间的ioc大于一个阈值即可，但是文字检测的目的是为了后续的文字识别，检测到半个字没意义。

因此，场景文字检测需要更高的精度，ctpn的主要基于RPN作文自检测，主要创新点有：

1. "we cast the problem of text detection into localizing a sequence of Fine-
scale text proposals."所谓的Fine-scale text proposals指的是宽度固定的一组proposals，最后的文本框由这些proposals连接构成。
2. "we propose an in-network recurrence mechanism that elegantly con-
nects sequential text proposals in the convolutional feature maps."即用了双向的LSTM捕获文本框的上下文信息。
3. "both methods are integrated seamlessly to meet the nature of text
sequence, resulting in a unified end-to-end trainable model."

****
# Connectionist Text Proposal Network
基于SSD，作者提出了一系列特殊的操作，使得网络能够检测不同方向的文字。

1. 提出了用四边形来表示任意方向的文字框(废话)
2. 将default-boxes回归到用四边形表示的bounding-boxes
3. 为了不仅保持横向default-boxes的密集性，而且还为了保证竖直方向的密集性，作者设置了default-boxes的竖直偏置
4. 作者在textboxes layer中设计了特殊的卷积(3*5)来替代传统的(3*3)卷积，以此来better handle text lines detecting
****
# Proposed network
整体网络结构如下图所示：

![](/img/ctpn/network.png)


原始CTPN只检测横向排列的文字。CTPN结构与Faster R-CNN基本类似，但是加入了LSTM层。假设输入 N Images：

1. 首先VGG提取特征，获得大小为 N \times C\times H\times W 的conv5 feature map。

2. 之后在conv5上做 3×3 的滑动窗口，即每个点都结合周围 3×3 区域特征获得一个长度为 3×3×C 的特征向量。输出 N \times9C\times H\times W 的feature map，该特征显然只有CNN学习到的空间特征。

3. 再将这个的feature map每一行都作为一个 T_{max}= W 的数据流，输入Bi-directional LSTM（双向LSTM），学习每一行的sequence feature。经过reshape后最终输出N \times256\times H\times W 特征，既包含空间特征，也包含了LSTM学习到的序列特征。

4. 再经过“FC”卷积层，变为 N \times512\times H\times W 的特征。

5. 最后经过类似Faster R-CNN的RPN网络，获得text proposals。

[详见](https://zhuanlan.zhihu.com/p/34757009)
****

