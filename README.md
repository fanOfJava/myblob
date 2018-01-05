## Welcome to Deep Learning Notes Pages

This is the main page for notes of papers on deep learning.

### Convolutional Networks
1. [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://github.com/fanOfJava/myblob/blob/master/papers/convolutional%20network/Very%20deep%20convolutional%20networks%20for%20large-scale%20image%20recognition.pdf)本篇论文主要探究卷积神经网络的深度对于大规模图像识别准确性的作用。本文主要的贡献在于通过增加大量使用3乘3卷积的网络层数（16-19层），来达到相比以前工作性能的大幅提高。卷积神经网络近年来在大规模图像和视频识别中取得了巨大的成功，这归功于大规模图片数据集的公开以及高性能的计算系统的开发。随着卷积神经网络在计算机视觉领域应用的增多，人们做了很多尝试，去提高原始AlexNet的性能。和以前的尝试不同，本论文中我们尝试着增加网络的***层数***来提高神经网络的性能。详见[Paper Notes](https://github.com/fanOfJava/myblob/blob/master/src/very_deep_convolutinal_networks.md)

2. [Going Deeper with Convolutions](https://github.com/fanOfJava/myblob/blob/master/papers/convolutional%20network/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)本篇论文提出一种由Inception架构而成的深度卷积神经网络，在ImageNet Large-Scale Visual Recognition Challenge 2014上分类和检测上都取得了最好的效果。Inception这种架构最大的优点在节省了网络的计算开销。通过精心的设计，我们在增加了**网络的深度和宽度的同时，保持计算量不变**。
深度学习以及神经网络快速发展，人们不再只关注更给力的硬件、更大的数据集、更大的模型，而是更在意新的idea、新的算法以及模型的改进。
一般来说，提升网络性能最直接的办法就是增加网络深度和宽度，这也就意味着巨量的参数。但是，巨量参数**容易产生过拟合**，也会大大**增加计算量**。
文章认为解决上述两个缺点的根本方法是将全连接甚至一般的卷积都转化为稀疏连接。早些的时候，为了打破网络对称性和提高学习能力，传统的网络都使用了随机稀疏连接。但是，计算机软硬件对非均匀稀疏数据的计算效率很差，所以在AlexNet中又重新启用了全连接层，目的是为了更好地优化并行运算。所以，现在的问题是有没有一种方法，既能**保持网络结构的稀疏性**，又能利用**密集矩阵的高计算性能**。大量的文献表明可以将稀疏矩阵聚类为较为密集的子矩阵来提高计算性能，据此论文提出了名为Inception的结构来实现此目的。详见[Paper Notes](https://github.com/fanOfJava/myblob/blob/master/src/going_deeper_with_convolutions.md)


