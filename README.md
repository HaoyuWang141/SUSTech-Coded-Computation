# Learning-based Non-linear Erasure Code for Distributed Inference

erasure code.pdf: 项目的总体方向

FusedLayer_test.py: 它对应于pdf中给出的第一步操作, 给定output的range, 计算input的range

其他可能需要的代码: 

+ Learning-Based_Coded_Computation 的仓库: 

  它对应于编码和解码的过程

  (1) https://github.com/Thesys-lab/learned-cc 

  (2) https://github.com/Thesyslab/parity-models

+ EdgeLD 的仓库:

  它对应于更后续的工作, 研究异质设备下, workload分配的部分

  https://github.com/fangvv/EdgeLD



## Dataset and Base Model

+ MNIST
  - [x] LeNet-5: [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)
+ CIFAR-10 / CIFAR-100
  - [ ] ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  - [ ] VGGNet: [Very Deep Convolutional Networks for Large-Scale Image Recognition ](https://arxiv.org/abs/1409.1556)
  - [ ] DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
  - [ ] EfficientNet: [ EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks ](https://arxiv.org/abs/1905.11946)
  - [ ] Wide ResNet: [Wide Residual Networks](https://arxiv.org/abs/1605.07146)



## Encoder and Decoder Architecture

### Encoder

input shape: $(k, a, a')$, output shape: $(r, a, a')$

- [x] MLP with two layers: $k a a' \times k a a'$, $kaa' \times raa'$
- [ ] CNN with ? layers

### Decoder

input shape: $(n, b, b')$, output shape: $(k, b, b')$

- [x] MLP with three layers: $nbb' \times kbb'$, $kbb' \times kbb'$, $kbb' \times kbb'$
- [ ] CNN with ? layers
