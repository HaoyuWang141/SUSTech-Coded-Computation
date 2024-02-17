"""
Split the original dataset into our form.
"""

from torchvision import datasets, transforms
from util.split_data import split_data
from base_model.LeNet5 import LeNet5
from base_model.VGG16 import VGG16
from base_model.ResNet18 import ResNet18
import torch

# TODO: change your base model and choose a trained model file

# LeNet5
# model = LeNet5(input_dim=(1, 28, 28), num_classes=10)
# base_model_path = "../base_model/LeNet5/MNIST/2023_11_28/model.pth"
# model.load_state_dict(torch.load(base_model_path))

# VGG16
# model = VGG16(input_dim=(3, 32, 32), num_classes=10)
# base_model_path = "../base_model/VGG16/CIFAR10/2023_11_28/model.pth"
# model.load_state_dict(torch.load(base_model_path))

# ResNet
model = ResNet18(input_dim=(3, 32, 32), num_classes=10)
base_model_path = "../base_model/ResNet18/CIFAR10/2023_12_30/model.pth"

# TODO: change your dataset and transform

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# CIFAR10
dataset_name = "CIFAR10"
test_dataset = datasets.CIFAR10(
    root="./", train=False, download=True, transform=transform
)
train_dataset = datasets.CIFAR10(
    root="./", train=True, download=True, transform=transform
)

# CIFAR100
# dataset_name = "CIFAR100"
# test_dataset = datasets.CIFAR100(root="./", train=False, download=True, transform=transform)
# train_dataset = datasets.CIFAR100(root="./", train=True, download=True, transform=transform)


# TODO: change your list of split numbers
split_nums = [1, 2]  # you may choose a subset from [1, 2, 4, ..]


print(f"Test dataset: {len(test_dataset)}")
print("image size: ", test_dataset[0][0].size())
print(f"Train dataset: {len(train_dataset)}")
print("image size: ", train_dataset[0][0].size())


# Split test dataset

for split_num in split_nums:
    split_data(
        layers=model.get_conv_segment(),
        dataset=test_dataset,
        split_num=split_num,
        train=False,
        output_file=f"./{dataset_name}/split/{split_num}/split_test_datasets.pt",
    )
    print("-" * 50)

# Split train dataset

for split_num in split_nums:
    split_data(
        layers=model.get_conv_segment(),
        dataset=train_dataset,
        split_num=split_num,
        train=True,
        output_file=f"./{dataset_name}/split/{split_num}/split_train_datasets.pt",
    )
    print("-" * 50)
