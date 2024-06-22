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
    "TASK": "IMAGENET10",
    "DATE": datetime.datetime.now().strftime("%Y_%m_%d"),
    "MODEL": "VGG16",
    "Division" : True
}

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),  # 首先将图像大小调整为256x256
    transforms.CenterCrop(224),  # 然后从中心裁剪出224x224的大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化，使用ImageNet的均值和标准差
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集（以CIFAR10为例，你需要替换为加载ImageNet10的逻辑）
train_dataset = datasets.ImageFolder(
   root='dataset/imagenet-10',
    transform=transform
)
test_dataset = datasets.ImageFolder(
    root='dataset/imagenet-10',
    transform=transform
)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Data is ready!")
print(f"当前任务为 {TASK_CONFIG['TASK']}")
print(f"当前模型为 {TASK_CONFIG['MODEL']}") 

K = 4
R = 2
N = K + R
Module = 5
original_data_shape = tuple(train_dataset[0][0].shape)
num_classes = 10
print(f"K: {K}")
print(f"R: {R}")
print(f"N: {N}")
print(f"data_shape: {original_data_shape}")
print(f"num_classes: {num_classes}")

import torch
from base_model.IMAGENET10_VGG16 import VGG16 as IMAGENET10_VGG16

# 引入 base model, 该model将在后续全部过程中使用
assert TASK_CONFIG["MODEL"] == "VGG16"


if TASK_CONFIG["TASK"] == "IMAGENET10":
    model = IMAGENET10_VGG16(input_dim=original_data_shape, num_classes=num_classes)
else:
    raise ValueError("Unknown Task")

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
    all_size = (param_size + buffer_size) / 1024 / 1024
    print("模型总大小为：{:.3f}MB".format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

getModelSize(model)
model.forward(torch.randn(1, *original_data_shape)).shape

# 读取模型
import torch.nn as nn
base_model_path = (
    f"./base_model/{TASK_CONFIG['MODEL']}/{TASK_CONFIG['TASK']}/model.pth"
)
print(f"base_model_path: {base_model_path}")

model.load_state_dict(torch.load(base_model_path, map_location=device))
getModelSize(model)
conv_segment = [None] * Module
for i in range(Module):
    conv_segment[i] = model.get_conv_segment(index = i + 1)
    print(conv_segment[i])
maxpool_segment = nn.MaxPool2d(kernel_size=2, stride=2)
fc_flatten = model.get_flatten()
fc_segment = model.get_fc_segment()
model.to(device)
model.eval()

print("Model is ready!")

x = torch.randn(1, *original_data_shape).to(device) # 生成一个随机输入
model.to(device) 
y = model.forward(x) # 进行一次前向传播
print(y.data)

shape_after_conv_segment = [None] * Module
shape_after_maxpool_segment = [None] * Module
print(f"Initial x shape: {x.shape}")
z = x
for i in range(Module):
    z = conv_segment[i](z)
    shape_after_conv_segment[i] = z.shape
    z = maxpool_segment(z)
    shape_after_maxpool_segment[i] = z.shape
    print(f"After conv_segment1, z shape: {shape_after_conv_segment[i]}")
    print(f"After maxpool_segment1, z shape: {shape_after_maxpool_segment[i]}")


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
    number_of_conv = 0
    for layer in conv_segment[i]:
        if isinstance(layer, nn.Conv2d):
            number_of_conv += 1
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

from encoder.conv_encoder import CatChannelConvEncoder
from decoder.conv_decoder import CatChannelConvDecoder


encoder = [None] * Module
decoder = [None] * Module
Encoder_Decoder_block = [0, 1]
for i in Encoder_Decoder_block:
    print(f"CASE: {i}")
    print(f"K : {K} R : {R} split_data_shape: {split_data_shape[i]}")
    print(f"N : {N} K : {K} split_conv_output_shape: {split_conv_output_shape[i]}")
    encoder[i] = CatChannelConvEncoder(num_in=K, num_out=R, in_dim=split_data_shape[i])
    getModelSize(encoder[i])
    decoder[i] = CatChannelConvDecoder(num_in=N, num_out=K, in_dim=split_conv_output_shape[i])
    getModelSize(decoder[i])

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

epoch_num = 3
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
def calc_result_endecoder(former, conv_segment, split_data_range, encoder, decoder, loss_num = 0):
    number_of_conv = 0
    for layer in conv_segment:
        if isinstance(layer, torch.nn.Conv2d):
            number_of_conv += 1
    images_list1 = []
    images_list2 = []
    for _1, _2, start, end in split_data_range:
        images_list1.append(former[:, :, :, start:end].clone())
        images_list2.append(former[:, :, :, start:end].clone())
    pad1 = (number_of_conv, 0, 0, 0, 0, 0, 0, 0)
    images_list1[0] = F.pad(images_list1[0], pad1, "constant", value=0)
    pad2 = (0, number_of_conv, 0, 0, 0, 0, 0, 0)
    images_list1[-1] = F.pad(images_list1[-1], pad2, "constant", value=0)
    # forward
    images_list2 += encoder(images_list1)
    output_list = []
    for j in range(N):
        output = images_list2[j]
        for layer in conv_segment:
            if isinstance(layer, torch.nn.Conv2d):
                if j == 0:
                    pad = (1, 0, 0, 0, 0, 0, 0, 0)
                    output = F.pad(output, pad, "constant", value=0)
                elif j == K - 1:
                    pad = (0, 1, 0, 0, 0, 0, 0, 0)
                    output = F.pad(output, pad, "constant", value=0)
                output = layer(output)
                output = output[:, :, :, 1:-1]
            else: 
                output = layer(output)
        output_list.append(output)
        
    if loss_num != 0:
        output_list = lose_something(output_list, loss_num)
    decoded_output_list = decoder(output_list) 
    output = torch.cat(decoded_output_list, dim=3) # 将数据拼接
    output = maxpool_segment(output)
    return output

save_dir = [None] * Module
encoder_path = [None] * Module
decoder_path = [None] * Module
for i in Encoder_Decoder_block:
    save_dir[i] = f"./save/{TASK_CONFIG['TASK']}/{TASK_CONFIG['MODEL']}/{TASK_CONFIG['DATE']}/K{K}-R{R}-conv-division-E-Dcoder-Num{i}/"
    encoder_path[i] = (
        save_dir[i]
        + f"encoder-task_{TASK_CONFIG['TASK']}-basemodel_{TASK_CONFIG['MODEL']}-K{K}-R{R}.pth"
    )
    decoder_path[i] = (
        save_dir[i]
        + f"decoder-task_{TASK_CONFIG['TASK']}-basemodel_{TASK_CONFIG['MODEL']}-K{K}-R{R}.pth"
    )

    print(f"save_dir: {save_dir[i]}")
    print(f"encoder_path: {encoder_path[i]}")
    print(f"decoder_path: {decoder_path[i]}")

import os;
print("original dir: ", os.getcwd())
for i in Encoder_Decoder_block:
    print(f"encoder_path: {encoder_path[i]}")
    print(f"decoder_path: {decoder_path[i]}")
    encoder[i].load_state_dict(torch.load(encoder_path[i], map_location=device))
    decoder[i].load_state_dict(torch.load(decoder_path[i], map_location=device))

import torch
from dataset.image_dataset import ImageDataset
from util.split_data import split_vector
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from random import randint, random
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
    merged_correct = 0
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
            original_correct += (predicted == labels).sum().item() # 没有分布式也没有encoder和decoder的准确率

            #Todo modify this part
            cur_lose = loss_num
            last_output = images
            tmp = 0
            for i in range(Module):
                if i not in Encoder_Decoder_block:
                    last_output = conv_segment[i](last_output)
                    last_output = maxpool_segment(last_output)
                else:
                    tmp = tmp + 1;
                    now_lose_num = random.randint(0, cur_lose)
                    if (tmp == len(Encoder_Decoder_block)):
                        now_lose_num = cur_lose
                        cur_lose = 0
                    else:
                        cur_lose = cur_lose - now_lose_num
                    last_output = calc_result_endecoder(last_output, conv_segment[i], split_data_range[i], encoder[i], decoder[i], now_lose_num)
                
            last_output = output.view(output.size(0), -1)
            _, predicted = torch.max(fc_segment(last_output).data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"样本总数: {total}")
    print(
        f"原始模型(conv+fc) -> 预测正确数: {original_correct}, 预测准确率: {(100 * original_correct / total):.2f}%"
    )
    print(
        f"使用Encoder和Decoder -> 预测正确数: {correct}, 预测准确率: {(100 * correct / total):.2f}%"
    )

# 训练集

for i in range(4):
    print(f"loss_num: {i}")
    evaluation(train_loader, i)

# 测试集
for i in range(4):
    print(f"loss_num: {i}")
    evaluation(test_loader, i)