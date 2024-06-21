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

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
# 设置数据增强和预处理
transform_train = transforms.Compose([
    transforms.Resize(256),  # 调整图像大小
    transforms.RandomCrop(224),  # 随机裁剪到224x224
    transforms.RandomHorizontalFlip(),  # 50%的概率水平翻转图像
    transforms.ToTensor(),  # 将图片转换为Tensor，并归一化到[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用ImageNet的均值和标准差
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # 中心裁剪到224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 假设你已经有了IMAGENET10的train_dataset和test_dataset
train_dataset = datasets.ImageFolder(root='dataset/imagenet-10', transform=transform_train)
test_dataset = datasets.ImageFolder(root='dataset/imagenet-10', transform=transform_test)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

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

from tqdm import tqdm
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