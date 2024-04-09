# CNN 经典模型

## LeNet

- **概述**：LeNet-5是最早的卷积神经网络之一，由Yann LeCun等人于1998年提出，主要用于手写数字识别。
- **论文**：Lecun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.



## AlexNet

- **概述**：AlexNet标志着深度学习在视觉识别任务中的重大突破，它在2012年的ImageNet竞赛中取得了压倒性胜利。
- **论文**：Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In *Advances in neural information processing systems* (pp. 1097-1105).



## VGGNet

- **概述**：VGGNet通过重复使用小卷积核（3x3）来构建深层网络，展示了通过增加深度可以显著提高网络性能的概念。
- **论文**：Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*.
- Visual Geometry Group 组



## GoogLeNet (Inception v1)

特点

+ Inception Structure
+ 1*1卷积核进行降维



## ResNet

- **概述**：ResNet通过引入残差学习的概念来解决深层网络训练中的梯度消失问题，使得网络能够达到前所未有的深度。
- **论文**：He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).



## DenseNet

- **概述**：DenseNet进一步改进了ResNet的想法，通过将每一层与前面所有层直接相连来加强特征的传递，极大提高了效率和效果。
- **论文**：Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4700-4708).