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
    "MODEL": "VGG16",
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
from base_model.IMAGENET10_VGG16 import VGG16 as IMAGENET10_VGG16
from base_model.CIFAR10_VGG16 import VGG16 as CIFAR10_VGG16

# 引入 base model, 该model将在后续全部过程中使用

if TASK_CONFIG["TASK"] == "IMAGENET10":
    model = IMAGENET10_VGG16(input_dim=original_data_shape, num_classes=num_classes)
if TASK_CONFIG["TASK"] == "CIFAR10":
    model = CIFAR10_VGG16(input_dim=original_data_shape, num_classes=num_classes)
else:
    raise ValueError("Unknown Task")

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

#!! 专门针对CIFAR10数据集的数据增强和预处理

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设置数据增强和预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 对图像进行随机裁剪，裁剪后图像大小为32x32，边缘填充4个像素
    transforms.RandomHorizontalFlip(),    # 50%的概率水平翻转图像
    transforms.RandomRotation(15),        # 随机旋转图像±15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机调整亮度、对比度、饱和度和色调
    transforms.ToTensor(),                # 将图片转换为Tensor，并归一化到[0,1]
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 标准化处理，使用CIFAR-10数据集的均值和标准差
])

# 测试集通常不应用太多变换，通常只进行标准化处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 创建数据加载器
train_dataset = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_dataset = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    tqdm_train_loader = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for idx, (data, target) in enumerate(tqdm_train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        tqdm_train_loader.set_postfix(loss=loss.item())
    train_loss /= idx + 1
    print(f"Train set: Average loss: {train_loss:.4f}, Accuracy: {(100 * correct / len(train_loader.dataset)):.2f}%")
    print('-'*50)

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        tqdm_test_loader = tqdm(test_loader, desc="Test")
        for idx, (data, target) in enumerate(tqdm_test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= idx + 1
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {(100 * correct / len(test_loader.dataset)):.2f}%")
    print('-'*50 + '\n')
    
BASE_MODEL_NUM_EPOCHS = 10
BASE_MODEL_LR = 1e-4
BASE_MODEL_MOMENTUM = 0.8
BASE_MODEL_WEIGHT_DECAY = 1e-6
# BASE_MODEL_CLIP_NORM = 1.0

optimizer = optim.Adam(model.parameters(), lr=BASE_MODEL_LR, weight_decay=BASE_MODEL_WEIGHT_DECAY)
# optimizer = optim.SGD(model.parameters(), lr=BASE_MODEL_LR, momentum=BASE_MODEL_MOMENTUM, weight_decay=BASE_MODEL_WEIGHT_DECAY)

criterion = CrossEntropyLoss()

model.to(device)
for epoch in range(BASE_MODEL_NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# Evaluation Process
# 测试循环
model.eval()  # 设置模型为评估模式

print("Train dataset:")

correct = 0
total = 0
with torch.no_grad():  # 在评估过程中不计算梯度
    for data, target in train_loader:
        # 将数据移动到设备上
        data, target = data.to(device), target.to(device)

        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"训练集-> 总量: {total}, 正确数量: {correct}, 准确率: {100 * correct / total}%")

print("Test dataset:")

correct = 0
total = 0
with torch.no_grad():  # 在评估过程中不计算梯度
    for data, target in test_loader:
        # 将数据移动到设备上
        data, target = data.to(device), target.to(device)

        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"测试集-> 总量: {total}, 正确数量: {correct}, 准确率: {100 * correct / total}%")

# 保存模型
now = datetime.datetime.now()
date = now.strftime("%Y_%m_%d")
filepath = f"base_model/{TASK_CONFIG['MODEL']}/{TASK_CONFIG['TASK']}/model.pth"
dirpath = os.path.dirname(filepath)
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
torch.save(model.state_dict(), filepath)