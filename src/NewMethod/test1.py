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
    print(f"split_data_range: {split_data_range[i]}")

    split_data_shapes[i] = [(tmp[0], tmp[1], tmp[3] - tmp[2],) for tmp in split_data_range[i]]
    print(f"split_data_shapes: {split_data_shapes[i]}")
    split_data_shape[i] = split_data_shapes[i][0]
    print(f"choose the first one as the split_data_shape: {split_data_shape[i]}")

for i in range(Module):
    print(f"CASE: {i}")
    if i == 0:
        current_original_data_shape = original_data_shape
    else:
        current_original_data_shape = shape_after_maxpool_segment[i - 1][1:]
    x = torch.randn(1, *current_original_data_shape).to(device)
    conv_segment[i].to(device)
    y = conv_segment[i](x)
    y = maxpool_segment(y)
    print(f"y.shape: {y.shape}")

    x_split = [x[:, :, :, _[2]:_[3]] for _ in split_data_range[i]]
    y_split = [conv_segment[i](_x) for _x in x_split]
    y_split = [maxpool_segment(_y) for _y in y_split]
    print(f"y_split.shape: {[tuple(_y.shape) for _y in y_split]}")
    print(f"y_split.shape: {[tuple(_y.shape) for _y in y_split]}")

    y_hat = torch.cat(y_split, dim=3)
    print(f"y_hat.shape: {y_hat.shape}")

    # |A-B| <= atol + rtol * |B|
    print(f"y和y_hat是否相等: {torch.allclose(y_hat, y, rtol=1e-08, atol=1e-05)}")

    diff = torch.abs(y_hat - y)
    epsilon = 0.0001
    print(f"y和y_hat是否相等: {torch.all(diff <= epsilon)}")

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

print(f"Train dataset: {len(train_dataset)}")
print("image size: ", train_dataset[0][0].size())

# 定义损失函数
criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

optimizer_encoder = [None] * Module
optimizer_decoder = [None] * Module
for i in range(Module):
    print(f"CASE: {i}")
    optimizer_encoder[i] = optim.Adam(encoder[i].parameters(), lr=1e-4, weight_decay=1e-6)
    optimizer_decoder[i] = optim.Adam(decoder[i].parameters(), lr=1e-4, weight_decay=1e-6)

    model.to(device)
    conv_segment[i].to(device)
    fc_segment.to(device)
    encoder[i].to(device)
    decoder[i].to(device)

    model.eval()
    conv_segment[i].eval()
    fc_segment.eval()
    encoder[i].train()
    decoder[i].train()

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False

    loss_list = [[] for _ in range(epoch_num)]

    for epoch in range(epoch_num):
        correct = 0
        correct_truth = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 把整一张完整的图片输入到模型中，得到ground_truth
            former = images
            for j in range(i):
                former = conv_segment[j](former)
                former = maxpool_segment(former)
            ground_truth = conv_segment[i](former)
            ground_truth = ground_truth.view(ground_truth.size(0), -1) # 将数据展平

            images_list = []
            for _1, _2, start, end in split_data_range[i]:
                images_list.append(former[:, :, :, start:end].clone())

            # forward
            images_list += encoder[i](images_list)
            output_list = []
            for j in range(N):
                output = conv_segment[i](images_list[j])
                output_list.append(output)
            # losed_output_list = lose_something(output_list, self.lose_device_index)
            decoded_output_list = decoder[i](output_list) 
            output = torch.cat(decoded_output_list, dim=3) # 将数据拼接
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

save_dir = f"./save/{TASK_CONFIG['TASK']}/{TASK_CONFIG['MODEL']}/{TASK_CONFIG['DATE']}/K{K}-R{R}-mlp-division/"
encoder_path = (
     save_dir
     + f"encoder-task_{TASK_CONFIG['TASK']}-basemodel_{TASK_CONFIG['MODEL']}-K{K}-R{R}.pth"
 )
decoder_path = (
     save_dir
     + f"decoder-task_{TASK_CONFIG['TASK']}-basemodel_{TASK_CONFIG['MODEL']}-K{K}-R{R}.pth"
 )

print(f"save_dir: {save_dir}")
print(f"encoder_path: {encoder_path}")
print(f"decoder_path: {decoder_path}")

import os


os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
os.makedirs(os.path.dirname(decoder_path), exist_ok=True)

torch.save(encoder.state_dict(), encoder_path)
torch.save(decoder.state_dict(), decoder_path)