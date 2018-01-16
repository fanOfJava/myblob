## Welcome to Deep Learning Notes Pages

This is the main page for notes of papers on deep learning.

![](/img/the_evolution_of_cnn.png)

### Convolutional Networks
1. [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://github.com/fanOfJava/myblob/blob/master/papers/convolutional%20network/Very%20deep%20convolutional%20networks%20for%20large-scale%20image%20recognition.pdf)本篇论文主要探究卷积神经网络的深度对于大规模图像识别准确性的作用。本文主要的贡献在于通过增加大量使用3乘3卷积的网络层数（16-19层），来达到相比以前工作性能的大幅提高。卷积神经网络近年来在大规模图像和视频识别中取得了巨大的成功，这归功于大规模图片数据集的公开以及高性能的计算系统的开发。随着卷积神经网络在计算机视觉领域应用的增多，人们做了很多尝试，去提高原始AlexNet的性能。和以前的尝试不同，本论文中我们尝试着增加网络的***层数***来提高神经网络的性能。详见[Paper Notes](https://github.com/fanOfJava/myblob/blob/master/src/very_deep_convolutinal_networks.md)

2. [Going Deeper with Convolutions](https://github.com/fanOfJava/myblob/blob/master/papers/convolutional%20network/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)本篇论文提出一种由Inception架构而成的深度卷积神经网络，在ImageNet Large-Scale Visual Recognition Challenge 2014上分类和检测上都取得了最好的效果。Inception这种架构最大的优点在节省了网络的计算开销。通过精心的设计，我们在增加了**网络的深度和宽度的同时，保持计算量不变**。
深度学习以及神经网络快速发展，人们不再只关注更给力的硬件、更大的数据集、更大的模型，而是更在意新的idea、新的算法以及模型的改进。
一般来说，提升网络性能最直接的办法就是增加网络深度和宽度，这也就意味着巨量的参数。但是，巨量参数**容易产生过拟合**，也会大大**增加计算量**。
文章认为解决上述两个缺点的根本方法是将全连接甚至一般的卷积都转化为稀疏连接。早些的时候，为了打破网络对称性和提高学习能力，传统的网络都使用了随机稀疏连接。但是，计算机软硬件对非均匀稀疏数据的计算效率很差，所以在AlexNet中又重新启用了全连接层，目的是为了更好地优化并行运算。所以，现在的问题是有没有一种方法，既能**保持网络结构的稀疏性**，又能利用**密集矩阵的高计算性能**。大量的文献表明可以将稀疏矩阵聚类为较为密集的子矩阵来提高计算性能，据此论文提出了名为Inception的结构来实现此目的。详见[Paper Notes](https://github.com/fanOfJava/myblob/blob/master/src/going_deeper_with_convolutions.md)


3. [Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift](https://github.com/fanOfJava/myblob/blob/master/papers/convolutional%20network/Batch%20Normalization.pdf)作者认为：网络训练过程中参数不断改变导致后续每一层输入的分布也发生变化，而学习的过程又要使每一层适应输入的分布，因此我们不得不降低学习率、小心地初始化。作者将分布发生变化称之为 **internal covariate shift**。大家应该都知道，我们一般在训练网络的时会将输入减去均值，还有些人甚至会对输入做白化等操作，目的是为了加快训练。**白化**的方式有好几种，常用的有**PCA白化**：即对数据进行PCA操作之后，在进行方差归一化。这样数据基本满足0均值、单位方差、弱相关性。作者首先考虑，对每一层数据都使用白化操作，但分析认为这是不可取的。因为白化需要计算**协方差矩阵、求逆**等操作，计算量很大，此外，反向传播时，**白化操作不一定可导**。于是，作者采用下面的Normalization方法。详见[Paper Notes](https://github.com/fanOfJava/myblob/blob/master/src/batch_normalization.md)

4. [googLeNet从V1-V4的演变过程](https://github.com/fanOfJava/myblob/blob/master/src/InceptionV1-V4.md)

5. [Deep Residual Learning for Image Recognition](https://github.com/fanOfJava/myblob/blob/master/papers/convolutional%20network/Deep%20residual%20learning%20for%20image%20recognition%20(2016)%2C%20K.%20He%20et%20al..pdf)
在图像分类中，往往要求神经网络需要比较深以达到好的效果，然而，随着网络层数的增加，**梯度消失/爆炸**的问题阻碍了模型的收敛。当然，可以通过标准的初始化和正则化层来基本解决解决梯度消失/爆炸的问题。然而随着神经网络深度的增加，却出现了精确度饱和的问题，而且会迅速的变差。这个实验结果让人一脸懵逼，直观上考虑两个神经网络，一个是浅层的神经网络，比如说20层，另一个与之对应的深层神经网络为56层，照理来说，56层的神经网络只要前面20层保持与浅层一样，后面36层，每层都做一个Indentity映射，这样就能保证训练误差不比20层的差。本文为了解决这个问题，引入了**残差模块**，并且根据此设计了**残差网络**。详见[Paper Notes](https://github.com/fanOfJava/myblob/blob/master/src/deep_residual_learning_for_image_recognition.md)

6. [Fully Convolutional Networks for Semantic Segmentation](https://github.com/fanOfJava/myblob/blob/master/papers/convolutional%20network/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
全卷积网络：从图像级理解到像素级理解。与物体分类要建立图像级理解任务不同的是，有些应用场景下要得到图像像素级别的分类结果，例如：1）语义级别图像分割(semantic image segmentation), 最终要得到对应位置每个像素的分类结果。2） 边缘检测, 相当于对每个像素做一次二分类（是边缘或不是边缘）。以语义图像分割为例，其目的是将图像分割为若干个区域, 使得语义相同的像素被分割在同意区域内。下图是一个语义图像分割的例子, 输入图像, 输出的不同颜色的分割区域表示不同的语义：背景、人和马。**转自**[ZhiHu](https://zhuanlan.zhihu.com/p/20872103?refer=dlclass)

### Object Detection
1. [Region-based Convolutional Networks for Accurate Object Detection and Segmentation](https://github.com/fanOfJava/myblob/blob/master/papers/object%20detection/Region-based%20convolutional%20networks%20for%20accurate%20object%20detection%20and%20segmentation%20(2016)%2C%20R.%20Girshick%20et%20al..pdf)《Rich feature hierarchies for Accurate Object Detection and Segmentation》，这篇文章的算法思想又被称之为：R-CNN（Regions with Convolutional Neural Network Features），是**物体检测**领域曾经获得state-of-art精度的经典文献。这篇paper的思想，改变了物体检测的总思路，现在好多文献关于深度学习的物体检测的算法，基本上都是继承了这个思想，学习经典算法，有助于我们以后搞物体检测的其它paper。详见[Paper Notes](https://github.com/fanOfJava/myblob/blob/master/src/region-based_convolutional_networks_for_accurate_object_detection_and_segmentation.md)


2. [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://github.com/fanOfJava/myblob/blob/master/papers/object%20detection/Spatial%20Pyramid%20Pooling%20in%20Deep%20Convolutional.pdf) 
本篇读书日记主要记录大神何凯明2014年的paper：《Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition》，这篇paper主要的创新点在于提出了空间金字塔池化，这个算法比R-CNN算法的速度快了n多倍。详见[Paper Notes](https://github.com/fanOfJava/myblob/blob/master/src/spatial_pyramid_pooling_in_deep_convolutional_networks_for_visual_recognition.md) 




