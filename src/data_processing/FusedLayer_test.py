"""
Author: 刘行
"""

import copy
import time
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # out = self.bn(out)
        out = self.relu(out)

        return out


class vgg16(nn.Module):
    def __init__(self, num_classes=1000):
        super(vgg16, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(64, 128, kernel_size=3, padding=1),
            BasicConv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(128, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(256, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layers = [*self.features]
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.layers = [*self.features]
        self.input_shape = 3, 224, 224
        self.depth = len(self.features)
        self.next = [*list(range(1, self.depth)), []]
        self.output_shapes = [(1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 112, 112), (1, 128, 112, 112),
                              (1, 128, 112, 112), (1, 128, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56),
                              (1, 256, 56, 56), (1, 256, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28),
                              (1, 512, 14, 14), (1, 512, 14, 14), (1, 512, 14, 14), (1, 512, 14, 14), (1, 512, 7, 7)]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_feature(self, x):
        return self.features(x)

    def forward_classifier(self, x):
        return self.classifier(torch.flatten(x))


model = vgg16()

output_shapes = model.output_shapes  # 18 layers in feature extraction


def generate_layerConfigs(layers: list):
    configs = []
    for layer in layers:
        if isinstance(layer, BasicConv2d):
            conv = layer.conv
            layer_config = {'type': 'basicConv', 'kernel_size': conv.kernel_size, 'stride': conv.stride,
                            'padding': conv.padding}  # , 'bn_args': (bn.weight, bn.bias, False, bn.momentum, bn.eps)}
        elif isinstance(layer, nn.Conv2d):
            layer_config = {'type': 'basicConv', 'kernel_size': layer.kernel_size, 'stride': layer.stride,
                            'padding': layer.padding}
        elif isinstance(layer, nn.ReLU):
            layer_config = {'type': 'relu', 'inplace': layer.inplace}
        elif isinstance(layer, nn.MaxPool2d):
            layer_config = {'type': 'maxpool', 'kernel_size': layer.kernel_size, 'stride': layer.stride,
                            'padding': layer.padding, 'ceil_mode': layer.ceil_mode}
        elif isinstance(layer, nn.Upsample):
            layer_config = {'type': 'upsample', 'scale_factor': layer.scale_factor}
        elif layer == 'concat':
            layer_config = {'type': layer}
        else:  # only given kinds of layers
            layer_config = None
            print('This type of layer is not supported yet')
        configs.append(layer_config)

    return configs


last_output_shape = output_shapes[-1]
length = last_output_shape[-1]

size = 1
range1 = (0, 4)
range2 = (3, 7)

layers = model.layers
layer_configs = generate_layerConfigs(layers)


def output_input(output_range: tuple, layer_config=None) -> tuple:
    o_s, o_e = output_range
    layer_type = layer_config['type']
    if layer_type in ['relu', 'concat']:  # most activation layers
        return output_range
    elif layer_type == 'upsample':
        scale_factor = layer_type['scale_factor']
        return round(o_s / scale_factor), round(o_e / scale_factor)
    elif layer_type in ('conv', 'basicConv', 'maxpool'):
        kernel_size, stride, padding = layer_config['kernel_size'], layer_config['stride'], layer_config['padding']
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if padding != 0:
            padding = padding[1]
        return o_s * stride[1] - padding, (o_e - 1) * stride[1] + kernel_size[1] - padding
    else:
        print('Unknown layer type')


def cal_ranges_backwards(layer_configs, output_shapes, last_output_range):
    '''
    given a layers chain, and the output range of the last layer, calculate the
    :param layer_configs: the layer configuration of the chain layers
    :param output_shapes: output shapes of the given layers
    :param last_output_range: the output range of the last layer
    :return: the input range of the first layer, and updated layer_configuration (and input range of intermediate layers)
    '''
    output_range = last_output_range
    first_input_range = None
    assert len(layer_configs) == len(output_shapes)
    new_layer_configs = copy.deepcopy(layer_configs)
    for l in range(len(layer_configs) - 1, -1, -1):  # consider there are just only conv and maxpool layers
        layer_config = new_layer_configs[l]  # layer config of the current layer
        # output_shape = output_shapes[l]
        # H = output_shape[-1]
        if l > 0:
            H = output_shapes[l - 1][-1]
        else:
            H = 224  # model.input_shape
        i_s, i_e = input_range = output_input(output_range, layer_config)  # [i_s, i_e)
        if layer_config['padding'] == 0:
            padding = (0, 0, 0, 0)
        else:
            if i_s < 0:
                upper_padding = -i_s
                i_s = 0
            else:
                upper_padding = 0
            if i_e > H:
                bottom_padding = i_e - H
                i_e = H
            else:
                bottom_padding = 0
            padding = (upper_padding, bottom_padding, *layer_config['padding'][-2:])
            input_range = (i_s, i_e)
        layer_config['padding'] = padding
        if l == 0:
            first_input_range = input_range
        else:
            output_range = input_range

    return first_input_range, new_layer_configs


input_range1, layer_configs1 = cal_ranges_backwards(layer_configs, output_shapes, range1)
print(f'output range: {range1} and input range: {input_range1}')
input_range2, layer_configs2 = cal_ranges_backwards(layer_configs, output_shapes, range2)
print(f'output range: {range2} and input range: {input_range2}')

# 3/7对应186/224???这也太多了

demo_input = torch.randn((1, 3, 224, 224))


# output = model.forward_feature(demo_input)
# print(output.shape)

def forward_fused_layer(x, layers, layer_configs):
    input = None
    out = None
    for l, layer in enumerate(layers):
        if l == 0:
            input = x
        layer_config = layer_configs[l]
        if isinstance(layer, BasicConv2d):
            weight = layer.conv.weight
            out = F.conv2d(F.pad(input, layer_config['padding']), weight, stride=layer_config['stride'])
            # out = F.batch_norm(out, torch.zeros(out.shape[1]), torch.ones(out.shape[1]), *self.operator['bn_args'])
            out = F.relu(out, inplace=True)
        elif isinstance(layer, nn.MaxPool2d):
            out = F.max_pool2d(F.pad(input, pad=layer_config['padding']), layer_config['kernel_size'],
                               stride=layer_config['stride'], padding=0, ceil_mode=layer_config['ceil_mode'])
        else:
            out = None
        input = out
    return out


consumption = time.time()

output = forward_fused_layer(demo_input[..., input_range2[0]:input_range2[1]], layers, layer_configs2)

consumption = time.time() - consumption

print(f'It takes {consumption} seconds to compute {size}/7 output of vgg16')

### check : forward(padded_input1 + padded_input1_input2) == forward(input1) + forward(input2) x

# padded_input1 = F.pad(demo_input[..., input_range1[0]: input_range1[1]], layer_configs1[0]['padding'])
# padded_input2 = F.pad(demo_input[..., input_range2[0]: input_range2[1]], layer_configs2[0]['padding'])
# print(padded_input1.shape, padded_input2.shape)
input1 = demo_input[..., input_range1[0]: input_range1[1]]
input2 = demo_input[..., input_range2[0]: input_range2[1]]

for lc in layer_configs1:
    if lc['padding'][-1] != 0: lc['padding'] = (0, 0, 0, 0)
for lc in layer_configs2:
    if lc['padding'][-1] != 0: lc['padding'] = (0, 0, 0, 0)

output1 = forward_fused_layer(input1, layers, layer_configs1)
output2 = forward_fused_layer(input2, layers, layer_configs2)
output_sum = output1 + output2

input12 = input1 + input2
output12 = forward_fused_layer(input12, layers, layer_configs1)

### forward(padded_input1 + padded_input1_input2) != forward(input1) + forward(input2)


print(output.shape)
