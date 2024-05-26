import os

print("original dir: ", os.getcwd())

if os.getcwd().endswith("NewMethod"):
    new_path = "../"
    os.chdir(new_path)
    print("changed dir: ", os.getcwd())
    
import torch
import torch.nn
from torch import optim
from torch.nn import CrossEntropyLoss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import datetime

TASK_CONFIG = {
    "TASK": "CIFAR10",
    "DATE": datetime.datetime.now().strftime("%Y_%m_%d"),
    "MODEL": "VGG10",
    "Division" : True
}

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设置数据转换
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# 设置数据集（训练集与测试集合）

"""
MNIST:
image: (1, 28, 28), label: (0-9)

FashionMNIST:
image: (1, 28, 28), label: (0-9)

CIFAR10:
image: (3, 32, 32), label: (0-9)
"""

print(f"当前任务为 {TASK_CONFIG['TASK']}")
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Data is ready!")

K = 4
R = 2
N = K + R
Module = 3
original_data_shape = tuple(train_dataset[0][0].shape)
num_classes = 10
print(f"K: {K}")
print(f"R: {R}")
print(f"N: {N}")
print(f"data_shape: {original_data_shape}")
print(f"num_classes: {num_classes}")

# 定义 base model
import torch

from base_model.MNIST_VGG10 import VGG10 as MNIST_VGG10
from base_model.Fashion_VGG10 import VGG10 as Fashion_VGG10
from base_model.CIFAR10_VGG10 import VGG10 as CIFAR10_VGG10
from base_model.Fashion_VGG10_Division import VGG10 as Fashion_VGG10_Division
from base_model.MNIST_VGG10_Division import VGG10 as MNIST_VGG10_Division
from base_model.CIFAR10_VGG10_Division import VGG10 as CIFAR10_VGG10_Division

# 引入 base model, 该model将在后续全部过程中使用
assert TASK_CONFIG["MODEL"] == "VGG10"

if TASK_CONFIG["TASK"] == "MNIST":
    if TASK_CONFIG["Division"] == False:
        model = MNIST_VGG10(input_dim=original_data_shape, num_classes=num_classes)
    else:
        model = MNIST_VGG10_Division(input_dim=original_data_shape, num_classes=num_classes)
elif TASK_CONFIG["TASK"] == "FashionMNIST":
    if TASK_CONFIG["Division"] == False:
        model = Fashion_VGG10(input_dim=original_data_shape, num_classes=num_classes)
    else:
        model = Fashion_VGG10_Division(input_dim=original_data_shape, num_classes=num_classes)
elif TASK_CONFIG["TASK"] == "CIFAR10":
    if TASK_CONFIG["Division"] == False:
        model = CIFAR10_VGG10(input_dim=original_data_shape, num_classes=num_classes)
    else:
        model = CIFAR10_VGG10_Division(input_dim=original_data_shape, num_classes=num_classes)
        print(f"input_dim: {original_data_shape} num_classes: {num_classes}")
else:
    raise ValueError("Unknown Task")

# 读取模型
import torch.nn as nn
base_model_path = (
    f"./base_model/{TASK_CONFIG['MODEL']}/{TASK_CONFIG['TASK']}/model.pth"
)
print(f"base_model_path: {base_model_path}")

model.load_state_dict(torch.load(base_model_path, map_location=device))
conv_segment = [None] * 3
conv_segment[0] = model.get_conv_segment(index = 1)
print(conv_segment[0])
conv_segment[1] = model.get_conv_segment(index = 2)
print(conv_segment[1])
conv_segment[2] = model.get_conv_segment(index = 3)
print(conv_segment[2])
maxpool_segment = nn.MaxPool2d(kernel_size=2, stride=2)
fc_flatten = model.get_flatten()
fc_segment = model.get_fc_segment()
model.to(device)
model.eval()

print("Model is ready!")

# 测试循环
model.eval()  # 设置模型为评估模式

correct = 0
total = 0
with torch.no_grad():  # 在评估过程中不计算梯度
    for data, target in train_loader:
        # 将数据移动到设备上
        data, target = data.to(device), target.to(device)
        output = model.forward(data, [conv_segment[0], maxpool_segment, conv_segment[1], maxpool_segment, conv_segment[2], maxpool_segment, 
                                      fc_flatten, fc_segment])
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"训练集-> 总量: {total}, 正确数量: {correct}, 准确率: {100 * correct / total}%")

correct = 0
total = 0
with torch.no_grad():  # 在评估过程中不计算梯度
    for data, target in test_loader:
        # 将数据移动到设备上
        data, target = data.to(device), target.to(device)
        output = model.forward(data, [conv_segment[0], maxpool_segment, conv_segment[1], maxpool_segment, conv_segment[2], maxpool_segment, 
                                      fc_flatten, fc_segment])
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"测试集-> 总量: {total}, 正确数量: {correct}, 准确率: {100 * correct / total}%")

x = torch.randn(1, *original_data_shape).to(device) # 生成一个随机输入
model.to(device) 
y = model.forward(x, [conv_segment[0], maxpool_segment, 
                      conv_segment[1], maxpool_segment, conv_segment[2], maxpool_segment ,fc_flatten, fc_segment]) # 进行前向传播
print(y.data)

shape_after_conv_segment = [None] * Module
shape_after_maxpool_segment = [None] * Module
print(f"Initial x shape: {x.shape}")
z = conv_segment[0](x)
shape_after_conv_segment[0] = z.shape
z = maxpool_segment(z)
shape_after_maxpool_segment[0] = z.shape
print(f"After conv_segment1, z shape: {shape_after_conv_segment[0]}")
print(f"After maxpool_segment1, z shape: {shape_after_maxpool_segment[0]}")

z = conv_segment[1](z)
shape_after_conv_segment[1] = z.shape
z = maxpool_segment(z)
shape_after_maxpool_segment[1] = z.shape
print(f"After conv_segment2, z shape: {shape_after_conv_segment[1]}")
print(f"After maxpool_segment1, z shape: {shape_after_maxpool_segment[1]}")

z = conv_segment[2](z)
shape_after_conv_segment[2] = z.shape
z = maxpool_segment(z)
shape_after_maxpool_segment[2] = z.shape
print(f"After conv_segment3, z shape: {shape_after_conv_segment[2]}")
print(f"After maxpool_segment1, z shape: {shape_after_maxpool_segment[2]}")


z = z.flatten(1) # 将数据展平
z = fc_segment(z) # 进行全连接层的计算 
print(z.data)   #用来验证model是否就是conv_segment + fc_segment
print(torch.allclose(y, z))

from util.util import cal_input_shape
from torch import nn
# encoder conv decoder maxpool
split_conv_output_shape = [None] * Module #经过了maxpool之后下一个encoder的输入大小
split_data_range = [None] * Module
split_data_shape = [None] * Module
split_data_shapes = [None] * Module
number_of_conv = 2
for i in range(Module):
    print(f"CASE: {i}")
    print(f"shape_after_conv_segment: {shape_after_conv_segment[i]}")
    assert shape_after_conv_segment[i][3] % K == 0

    split_conv_output_shape[i] = (
        shape_after_conv_segment[i][1],
        shape_after_conv_segment[i][2],
        shape_after_conv_segment[i][3] // K,
    ) #分割输出的大小
    print(f"split_conv_output_shape: {split_conv_output_shape[i]}") #Decoder的input大小

    conv_segment[i].to('cpu')
    conv_segment[i].train()
    current_original_data_shape = [None]
    if i == 0:
        current_original_data_shape = original_data_shape
    else:
        current_original_data_shape = shape_after_maxpool_segment[i - 1][1:]

    print(f"current_original_data_shape: {current_original_data_shape}") #Encoder的未分割的input大小
    split_data_range[i] = cal_input_shape( #计算Encoder的分割的input大小
        model=conv_segment[i],
        original_input_shape=current_original_data_shape,
        original_output_shape=shape_after_conv_segment[i][1:], #通过输出的大小回推输入的大小(只包括conv不包括maxpool)
        split_num=K,
    )
    for j in range(K):
        split_data_range[i][j] = (split_data_range[i][j][0], split_data_range[i][j][1], max(split_data_range[i][j][2] - number_of_conv, 0), 
                           min(split_data_range[i][j][3] + number_of_conv, current_original_data_shape[2]))
    print(f"split_data_range: {split_data_range[i]}")

    split_data_shapes[i] = [(tmp[0], tmp[1], tmp[3] - tmp[2] + number_of_conv,) for tmp in split_data_range[i]]
    print(f"split_data_shapes: {split_data_shapes[i]}")
    split_data_shape[i] = split_data_shapes[i][0]
    print(f"choose the first one as the split_data_shape: {split_data_shape[i]}")

from encoder.mlp_encoder_division import MLPEncoder
from decoder.mlp_decoder_division import MLPDecoder


encoder = [None] * Module
decoder = [None] * Module

for i in range(Module):
    print(f"CASE: {i}")
    print(f"split_data_shape: {split_data_shape[i]}")
    print(f"split_conv_output_shape: {split_conv_output_shape[i]}")
    encoder[i] = MLPEncoder(num_in=K, num_out=R, in_dim=split_data_shape[i])
    decoder[i] = MLPDecoder(num_in=N, num_out=K, in_dim=split_conv_output_shape[i])

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024
    print("模型总大小为：{:.3f}KB".format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

for i in range(Module):
    print(f"CASE: {i}")
    getModelSize(encoder[i])
    getModelSize(decoder[i])

epoch_num = 10
print(f"epoch_num: {epoch_num}")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
print(f"Train dataset: {len(train_dataset)}")
print("image size: ", train_dataset[0][0].size())

# 定义损失函数
criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

optimizer_encoder = [None] * Module
optimizer_decoder = [None] * Module
model.to(device)
fc_segment.to(device)
fc_segment.eval()
model.eval()
start_time = datetime.datetime.now()
for i in range(2,3):
    print(f"CASE: {i}")
    optimizer_encoder[i] = optim.Adam(encoder[i].parameters(), lr=1e-4, weight_decay=1e-6)
    optimizer_decoder[i] = optim.Adam(decoder[i].parameters(), lr=1e-4, weight_decay=1e-6)

    conv_segment[i].to(device)
    encoder[i].to(device)
    decoder[i].to(device)

    conv_segment[i].eval()
    encoder[i].train()
    decoder[i].train()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False

    loss_list = [[] for _ in range(epoch_num)]
    for epoch in range(epoch_num):
        train_loader_tqdm = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epoch_num}",
            bar_format="{l_bar}{bar:20}{r_bar}",
        )
        correct = 0
        correct_truth = 0
        total = 0
        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device)

            # 把整一张完整的图片输入到模型中，得到ground_truth
            former = images
            for j in range(i):
                former = conv_segment[j](former)
                former = maxpool_segment(former)
            ground_truth = conv_segment[i](former)
            ground_truth = maxpool_segment(ground_truth)
            ground_truth = ground_truth.view(ground_truth.size(0), -1) # 将数据展平
            images_list = []
            for _1, _2, start, end in split_data_range[i]:
                images_list.append(former[:, :, :, start:end].clone())
            pad1 = (number_of_conv, 0, 0, 0, 0, 0, 0, 0)
            images_list[0] = F.pad(images_list[0], pad1, "constant", value=0)
            pad2 = (0, number_of_conv, 0, 0, 0, 0, 0, 0)
            images_list[-1] = F.pad(images_list[-1], pad2, "constant", value=0)
            # forward
            images_list += encoder[i](images_list)
            output_list = []
            for j in range(N):
                output = conv_segment[i](images_list[j])
                output = output[:, :, :, number_of_conv:-number_of_conv]
                output_list.append(output)
            # losed_output_list = lose_something(output_list, self.lose_device_index)
            decoded_output_list = decoder[i](output_list) 
            output = torch.cat(decoded_output_list, dim=3) # 将数据拼接
            output = maxpool_segment(output)
            output = output.view(output.size(0), -1) # 将数据展平
            
            loss = criterion(output, ground_truth)
            # loss = criterion2(fc_segment(output), fc_segment(ground_truth))

            loss_list[epoch].append(loss.item())

            # backward
            optimizer_encoder[i].zero_grad()
            optimizer_decoder[i].zero_grad()
            loss.backward()
            optimizer_encoder[i].step()
            optimizer_decoder[i].step()
            _, predicted_truth = torch.max(fc_segment(ground_truth.data), 1)
            _, predicted = torch.max(fc_segment(output).data, 1)
            correct += (predicted == labels).sum().item()
            correct_truth += (predicted_truth == labels).sum().item()
            total += labels.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        print(f"Train Accuracy: {100 * correct / total}%")
        print(f"Original Accuracy: {100 * correct_truth / total}%")
end_time = datetime.datetime.now()
print(f"Time cost for Training: {end_time - start_time}")
def lose_something(output_list, lose_num):
    if lose_num == 0:
        return output_list
    
    lose_index = torch.randperm(len(output_list))[:lose_num]
    losed_output_list = []

    for i in range(len(output_list)):

        if i in lose_index:

            losed_output_list.append(torch.zeros_like(output_list[i]))
        else:

            losed_output_list.append(output_list[i])
    return losed_output_list

import torch
from dataset.image_dataset import ImageDataset
from util.split_data import split_vector

fc_segment.to(device)
model.to(device)
for i in range(Module):
    conv_segment[i].to(device)
    encoder[i].to(device)
    decoder[i].to(device)

fc_segment.eval()
model.eval()
for i in range(Module):
    conv_segment[i].eval()
    encoder[i].eval()
    decoder[i].eval()

def evaluation(loader, loss_num):
    original_correct = 0
    merge_correct = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            output = images
            for i in range(Module):
                output = conv_segment[i](output)
                output = maxpool_segment(output)
            output = output.view(output.size(0), -1)
            output = fc_segment(output)
            _, predicted = torch.max(output.data, 1)
            merge_correct += (predicted == labels).sum().item()

            #Todo modify this part
            last_output = images
            for i in range(Module - 1):
                last_output = conv_segment[i](last_output)
                last_output = maxpool_segment(last_output)
            for j in range(Module - 1, Module):
                images_list = []
                for _1, _2, start, end in split_data_range[j]:
                    images_list.append(last_output[:, :, :, start:end].clone())
                pad1 = (number_of_conv, 0, 0, 0, 0, 0, 0, 0)
                images_list[0] = F.pad(images_list[0], pad1, "constant", value=0)
                pad2 = (0, number_of_conv, 0, 0, 0, 0, 0, 0)
                images_list[-1] = F.pad(images_list[-1], pad2, "constant", value=0)
                imageDataset_list = [
                    ImageDataset(images) for images in images_list + encoder[j](images_list)
                ]
                output_list = []
                for i in range(N):
                    imageDataset = imageDataset_list[i]
                    output = conv_segment[j](imageDataset.images)
                    output = output[:, :, :, number_of_conv:-number_of_conv]
                    output_list.append(output)
                losed_output_list = lose_something(output_list, loss_num)
                decoded_output_list = decoder[j](losed_output_list)
                output = torch.cat(decoded_output_list, dim=3)
                output = maxpool_segment(output)
                last_output = output
                
            last_output = output.view(output.size(0), -1)
            _, predicted = torch.max(fc_segment(last_output).data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"样本总数: {total}")
    print(
        f"原始模型(conv+fc) -> 预测正确数: {merge_correct}, 预测准确率: {(100 * merge_correct / total):.2f}%"
    )
    print(
        f"使用Encoder和Decoder -> 预测正确数: {correct}, 预测准确率: {(100 * correct / total):.2f}%"
    )

# 训练集
start_time = datetime.datetime.now()
for i in range(N + 1):
    print(f"loss_num: {i}")
    evaluation(train_loader, i)

# 测试集
for i in range(N + 1):
    print(f"loss_num: {i}")
    evaluation(test_loader, i)
end_time = datetime.datetime.now()
print(f"Time cost for Evaluation: {end_time - start_time}")