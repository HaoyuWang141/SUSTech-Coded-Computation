import os
import torch
import torch.nn
from torch import optim
from torch.nn import CrossEntropyLoss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm
import datetime

TASK_CONFIG = {
    "TASK": "CIFAR10",
    "DATE": datetime.datetime.now().strftime("%Y_%m_%d"),
    "MODEL": "VGG10",
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

# train_dataset = datasets.MNIST(
#     root="./data", train=True, download=True, transform=transform
# )
# test_dataset = datasets.MNIST(
#     root="./data", train=False, download=True, transform=transform
# )
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
original_data_shape = tuple(train_dataset[0][0].shape)
num_classes = 10
print(f"K: {K}")
print(f"R: {R}")
print(f"N: {N}")
print(f"data_shape: {original_data_shape}")
print(f"num_classes: {num_classes}")

# 定义 base model

from base_model.MNIST_VGG10 import VGG10 as MNIST_VGG10
from base_model.Fashion_VGG10 import VGG10 as Fashion_VGG10
from base_model.CIFAR10_VGG10 import VGG10 as CIFAR10_VGG10

# 引入 base model, 该model将在后续全部过程中使用
assert TASK_CONFIG["MODEL"] == "VGG10"

if TASK_CONFIG["TASK"] == "MNIST":
    model = MNIST_VGG10(input_dim=original_data_shape, num_classes=num_classes)
elif TASK_CONFIG["TASK"] == "FashionMNIST":
    model = Fashion_VGG10(input_dim=original_data_shape, num_classes=num_classes)
elif TASK_CONFIG["TASK"] == "CIFAR10":
    model = CIFAR10_VGG10(input_dim=original_data_shape, num_classes=num_classes)
else:
    raise ValueError("Unknown Task")

# 读取模型
base_model_path = (
    f"./base_model/{TASK_CONFIG['MODEL']}/{TASK_CONFIG['TASK']}/model.pth"
)
print(f"base_model_path: {base_model_path}")

model.load_state_dict(torch.load(base_model_path, map_location=device))
conv_segment = model.get_conv_segment()
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

        output = model(data)
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

        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"测试集-> 总量: {total}, 正确数量: {correct}, 准确率: {100 * correct / total}%")

x = torch.randn(1, *original_data_shape).to(device)
model.to(device)
y = model(x)
print(y.data)

z = conv_segment(x)
z = z.flatten(1)
z = fc_segment(z)
print(z.data)
print(torch.allclose(y, z))

from util.util import cal_input_shape


conv_output_shape = model.calculate_conv_output(input_dim=original_data_shape)
print(f"conv_output_shape: {conv_output_shape}")
assert conv_output_shape[2] % K == 0

split_conv_output_shape = (
    conv_output_shape[0],
    conv_output_shape[1],
    conv_output_shape[2] // K,
)
print(f"split_conv_output_shape: {split_conv_output_shape}")

conv_segment.to('cpu')
conv_segment.train()
split_data_range = cal_input_shape(
    model=conv_segment,
    original_input_shape=original_data_shape,
    original_output_shape=conv_output_shape,
    split_num=K,
)
print(f"split_data_range: {split_data_range}")

# print(conv_segment)
print(
    f"split_conv_output_data_shape from split_data_shape: {[tuple(conv_segment(torch.randn(1, _[0], _[1], _[3] - _[2])).shape) for _ in split_data_range]}"
)

split_data_shapes = [
    (
        _[0],
        _[1],
        _[3] - _[2],
    )
    for _ in split_data_range
]
print(f"split_data_shapes: {split_data_shapes}")
print(type(split_data_shapes))
split_data_shape = split_data_shapes[0]
print(f"choose the first one as the split_data_shape: {split_data_shape}")

x = torch.randn(1, *original_data_shape).to(device)
conv_segment.to(device)
y = conv_segment(x)
print(f"y.shape: {y.shape}")

x_split = [x[:, :, :, _[2]:_[3]] for _ in split_data_range]
y_split = [conv_segment(_x) for _x in x_split]
print(f"y_split.shape: {[tuple(_y.shape) for _y in y_split]}")
print(f"y_split.shape: {[tuple(_y.shape) for _y in y_split]}")

y_hat = torch.cat(y_split, dim=3)
print(f"y_hat.shape: {y_hat.shape}")

# |A-B| <= atol + rtol * |B|
print(f"y和y_hat是否相等: {torch.allclose(y_hat, y, rtol=1e-08, atol=1e-05)}")

diff = torch.abs(y_hat - y)
epsilon = 0.0001
print(f"y和y_hat是否相等: {torch.all(diff <= epsilon)}")
# print(torch.allclose(y_split[0], y[:, :, :, 0:5]))
# print(torch.allclose(y_split[1], y[:, :, :, 5:10]))
# print(torch.allclose(y_split[2], y[:, :, :, 10:15]))
# print(torch.allclose(y_split[3], y[:, :, :, 15:20]))

# print(y[0][0][0] == y_hat[0][0][0])
# print(y[0][0][0])
# print(y_hat[0][0][0])
# y = x
# y_split = x_split
# for layer in conv_segment:
#     print(layer)
#     y = layer(y)
#     y_split = [layer(_y) for _y in y_split]
#     print(f"y.shape: {y.shape}")
#     print(f"y_split.shape: {[tuple(_.shape) for _ in y_split]}")
#     print(y[0][0][0])
#     print(y_split[0][0][0][0])
#     print(y_split[1][0][0][0])
#     print(y_split[2][0][0][0])
#     print(y_split[3][0][0][0])
from encoder.mlp_encoder import MLPEncoder
from encoder.conv_encoder import CatChannelConvEncoder, CatBatchSizeConvEncoder
from decoder.mlp_decoder import MLPDecoder
from decoder.conv_decoder import CatChannelConvDecoder, CatBatchSizeConvDecoder

print(f"split_data_shape: {split_data_shape}")
print(f"split_conv_output_shape: {split_conv_output_shape}")

# encoder = MLPEncoder(num_in=K, num_out=R, in_dim=split_data_shape)
decoder = MLPDecoder(num_in=N, num_out=K, in_dim=split_conv_output_shape)

encoder = CatChannelConvEncoder(num_in=K, num_out=R, in_dim=split_data_shape)
# decoder = CatChannelConvDecoder(num_in=N, num_out=K, in_dim=split_conv_output_shape)

# encoder = CatBatchSizeConvEncoder(num_in=K, num_out=R, in_dim=split_data_shape)
# decoder = CatBatchSizeConvDecoder(num_in=N, num_out=K, in_dim=split_conv_output_data_shape)
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

getModelSize(encoder)
getModelSize(decoder)
print()

epoch_num = 10
print(f"epoch_num: {epoch_num}")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

print(f"Train dataset: {len(train_dataset)}")
print("image size: ", train_dataset[0][0].size())

# 定义损失函数
criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

# optimizer_encoder = optim.SGD(encoder.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
# optimizer_decoder = optim.SGD(decoder.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

optimizer_encoder = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-6)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=1e-4, weight_decay=1e-6)

model.to(device)
conv_segment.to(device)
fc_segment.to(device)
encoder.to(device)
decoder.to(device)

model.eval()
conv_segment.eval()
fc_segment.eval()
encoder.train()
decoder.train()

model.eval()
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

        # split image tensor(64, 3, 32, 32) -> [tensor(64, 3, 32, 8) * K]
        images_list = []
        for _1, _2, start, end in split_data_range:
            images_list.append(images[:, :, :, start:end].clone())

        pad = (0, 7, 0, 0)
        images_list[-1] = F.pad(images_list[-1], pad, "constant", value=0)

        ground_truth = conv_segment(images)
        ground_truth = ground_truth.view(ground_truth.size(0), -1)

        # forward
        images_list += encoder(images_list)
        output_list = []
        for i in range(N):
            output = conv_segment(images_list[i])
            output_list.append(output)
        # losed_output_list = lose_something(output_list, self.lose_device_index)
        decoded_output_list = decoder(output_list)
        output = torch.cat(decoded_output_list, dim=3)
        output = output.view(output.size(0), -1)

        loss = criterion(output, ground_truth)
        # loss = criterion2(fc_segment(output), fc_segment(ground_truth))

        loss_list[epoch].append(loss.item())

        # backward
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        # calculate accuracy
        _, predicted = torch.max(fc_segment(output).data, 1)
        _, predicted_truth = torch.max(fc_segment(ground_truth.data), 1)
        # print(predicted)
        # print(predicted_truth)
        # print(labels)
        correct += (predicted == labels).sum().item()
        correct_truth += (predicted_truth == labels).sum().item()
        total += labels.size(0)

        train_loader_tqdm.set_postfix(loss=loss.item())

    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    print(f"Train Accuracy: {100 * correct / total}%")
    print(f"Original Accuracy: {100 * correct_truth / total}%")
    # 27%
    # 10%