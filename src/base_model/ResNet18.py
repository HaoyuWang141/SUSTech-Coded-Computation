import math
import torch
import torch.nn as nn
from base_model.BaseModel import BaseModel


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def get_sequence(self):
        # Get sequence of layers except the downsample layer
        return nn.Sequential(
            self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu
        )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def get_sequence(self):
        # Get sequence of layers except the downsample layer
        return nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
            self.conv3,
            self.bn3,
            self.relu,
        )


class ResNet18(BaseModel):
    def __init__(
        self,
        input_dim: tuple[int],
        num_classes: int,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        size_for_cifar=True,
    ):
        # Argument `input_dim` is only used to check whether 'size_for_cifar' is True.

        self.blk = block
        self.inplanes = 64
        super(ResNet18, self).__init__()
        input_dim = tuple(input_dim)
        num_classes = int(num_classes)
        size_for_cifar = True if input_dim[0] == 3 else False  #
        self.size_for_cifar = size_for_cifar
        num_channels = 3 if size_for_cifar else 1

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        if size_for_cifar:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            # self.avgpool = nn.AvgPool2d(4)
            # self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.fc = nn.Linear(512 * block.expansion * 4 * 4, num_classes)
        else:
            # self.avgpool = nn.AvgPool2d(7)
            # self.fc = nn.Linear(256 * block.expansion, num_classes)
            self.fc = nn.Linear(256 * block.expansion * 7 * 7, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # We only use the final layer if `x` started as (-1, 3, 32, 32)
        if self.size_for_cifar:
            x = self.layer4(x)

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_conv_segment(self) -> nn.Sequential:
        if self.size_for_cifar:
            raw = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                *self.layer1,
                *self.layer2,
                *self.layer3,
                *self.layer4
            )
        else:
            raw = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                *self.layer1,
                *self.layer2,
                *self.layer3
            )
        # layers = []
        # for layer in raw:
        #     if isinstance(layer, BasicBlock) or isinstance(layer, Bottleneck):
        #         for sub_layer in layer.get_sequence():
        #             layers.append(sub_layer)
        #     else:
        #         layers.append(layer)
        # return nn.Sequential(*layers)
        return raw

    def get_fc_segment(self) -> nn.Sequential:
        # return nn.Sequential(self.avgpool, self.fc)
        return nn.Sequential(self.fc)

    def calculate_conv_output(self, input_dim: tuple[int]) -> tuple[int, int, int]:
        # Assuming input_dim is a tuple (channels, height, width)
        channels, height, width = input_dim

        def conv2d_out_size(size, kernel_size=3, stride=1, padding=1):
            # DEPRECATED
            # This method is useless because the input size of fc_segment is already computed in the original code.
            return (
                (size[0] - kernel_size + 2 * padding) // stride + 1,
                (size[1] - kernel_size + 2 * padding) // stride + 1,
            )

        def maxpool2d_out_size(size, kernel_size=2, stride=2, padding=0):
            # DEPRECATED
            # This method is useless because the input size of fc_segment is already computed in the original code.
            return (
                (size[0] - kernel_size + 2 * padding) // stride + 1,
                (size[1] - kernel_size + 2 * padding) // stride + 1,
            )

        if self.size_for_cifar:
            # return (512 * self.blk.expansion, 1, 1)
            return (512 * self.blk.expansion, 4, 4)
        else:
            # return (256 * self.blk.expansion, 1, 1)
            return (256 * self.blk.expansion, 7, 7)


if __name__ == "__main__":
    # Example usage

    num_classes = 10

    # input_dim = (1, 28, 28)
    # x = torch.randn(1, 1, 28, 28)
    input_dim = (3, 32, 32)
    x = torch.randn(1, 3, 32, 32)

    model = ResNet18(input_dim, num_classes)
    # model = ResNet18(input_dim, num_classes, block=Bottleneck)

    # print(model)
    # print(model.get_conv_segment())

    conv_segment = model.get_conv_segment()
    fc_segment = model.get_fc_segment()

    y = model(x)
    print("y", y.shape)
    print("y", y[0][0])

    print("-" * 100)
    # for index, layer in enumerate(conv_segment):
    #     print(index)
    #     print(layer)
    #     x = layer(x)
    #     print(x.shape)
    #     print(x)
        
    #     if index == 3:
    #         break
    
    y1 = conv_segment(x)
    print("y1", y1.shape)
    print('conv输出:')
    print(y1)
    y1 = y1.view(y1.size(0), -1)
    y2 = fc_segment(y1)
    print("y2", y2.shape)
    print(y)
    print(y2)
    
    # print(model)
    # print('-----------------')
    # print(model.get_conv_segment())
