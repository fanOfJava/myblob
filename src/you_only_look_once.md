# You Only Look Once:Unified, Real-Time Object Detection

作者提出了一种新的物体检测方法YOLO。YOLO之前的物体检测方法主要是通过region proposal产生大量的可能包含待检测物体的 potential bounding box，再用分类器去判断每个 bounding box里是否包含有物体，以及物体所属类别的probability或者 confidence，如R-CNN,Fast-R-CNN,Faster-R-CNN等。

YOLO不同于这些物体检测方法，它将物体检测任务当做一个regression问题来处理，使用一个神经网络，直接从一整张图像来预测出bounding box 的坐标、box中包含物体的置信度和物体的probabilities。因为YOLO的物体检测流程是在一个神经网络里完成的，所以可以end to end来优化物体检测性能。

YOLO检测物体的速度很快，标准版本的YOLO在Titan X 的 GPU 上能达到45 FPS。网络较小的版本Fast YOLO在保持mAP是之前的其他实时物体检测器的两倍的同时，检测速度可以达到155 FPS。

相较于其他的state-of-the-art 物体检测系统，YOLO在物体定位时更容易出错，但是在背景上预测出不存在的物体（false positives）的情况会少一些。而且，YOLO比DPM、R-CNN等物体检测系统能够学到更加抽象的物体的特征，这使得YOLO可以从真实图像领域迁移到其他领域，如艺术。